"""Optional Chinese notebook UI. No search runs on import or display."""

from datetime import datetime
from fractions import Fraction
from html import escape
import json
from pathlib import Path
from uuid import uuid4

from dataclasses import dataclass

from .domain import CountMode
from .label_catalog import (
    FreshCatalogResult,
    LabelCatalogBundle,
    TopologyLabelCatalog,
    catalog_from_audit,
    read_label_catalog,
    run_label_catalog,
    write_label_catalog,
)
from .physical_cards import build_physics_card, render_card_markdown
from .physical_labels import NormalizationResult


REASONS = {
    'dependent_invariants': '双线不变质量之间存在依赖关系',
    'empty_composite': '双线动量为空',
    'singleton_composite': '双线只包含一个粒子，质量固定',
    'fixed_total_composite': '双线包含全部粒子，总质量固定',
    'negative_composite_coefficient': '双线动量中出现负的粒子系数',
    'composite_coefficient_gt_one': '双线动量重复计入同一粒子',
    'declared_double_momentum_mismatch': '声明的双线粒子集合与守恒解不同',
}
STATES = {'complete': '搜索完成', 'incomplete': '搜索未完成', 'failed': '运行失败'}

CATALOG_ARTIFACT = 'label_catalog'


@dataclass(frozen=True, slots=True)
class BrowserSource:
    """One selected file and its converted view; only this file is loaded."""

    path: Path
    source_kind: str
    mode: CountMode
    audit: dict | None
    catalog: LabelCatalogBundle | None
    loop_count: int | None
    loaded_from_cache: bool = False


class _BrowserSourceCache:
    def __init__(self):
        self._items = {}

    def clear(self):
        self._items.clear()

    def load(self, path: Path, mode: CountMode) -> BrowserSource:
        path = Path(path)
        stat = path.stat()
        key = (str(path.resolve()), stat.st_mtime_ns, stat.st_size, mode.value, CATALOG_ARTIFACT)
        cached = self._items.get(key)
        if cached is not None:
            return BrowserSource(
                cached.path, cached.source_kind, cached.mode, cached.audit, cached.catalog,
                cached.loop_count, True
            )
        payload = json.loads(path.read_text(encoding='utf-8'))
        if payload.get('artifact_type') == CATALOG_ARTIFACT:
            catalog = read_label_catalog(path)
            loop_count = next(
                (item.search_evidence.loop_count for item in catalog.catalogs
                 if item.search_evidence.loop_count is not None), None
            )
            source = BrowserSource(path, CATALOG_ARTIFACT, mode, None, catalog, loop_count)
        else:
            if not isinstance(payload, dict):
                raise ValueError('结果文件顶层必须是 JSON 对象')
            catalog = catalog_from_audit(
                payload,
                mode=mode,
                source_info={'source_path': str(path), 'source_kind': 'legacy_audit'},
            )
            loop_count = (payload.get('topology_report') or {}).get('loop_count')
            source = BrowserSource(path, 'legacy_audit', mode, payload, catalog, loop_count)
        self._items[key] = source
        return source


_SOURCE_CACHE = _BrowserSourceCache()


def load_browser_source(path: str | Path, *, mode: CountMode | str = CountMode.FIXED_LABELS) -> BrowserSource:
    """Load one selected legacy audit or label-catalog artifact."""

    selected = mode if isinstance(mode, CountMode) else CountMode(mode)
    return _SOURCE_CACHE.load(Path(path), selected)


def clear_browser_source_cache() -> None:
    """Forget file metadata and converted bundles after an explicit refresh."""

    _SOURCE_CACHE.clear()


def _catalog_status(catalog: TopologyLabelCatalog) -> str:
    if catalog.identity_count:
        return '存在有效方案；已找全' if catalog.enumeration_complete else '存在有效方案；尚未找全'
    if catalog.existence == 'no' and catalog.enumeration_complete:
        return '无有效标记（搜索与记录已完整）'
    return '未计算／部分目录'


def reason(code):
    return REASONS.get(code, '未翻译的原因：' + str(code))


def stage(report):
    if report.get('status') != 'complete':
        return '待完成／运行失败'
    if report.get('valid_results'):
        return '有效拓扑'
    if report.get('counts', {}).get('momentum_valid', 0) > 0:
        return '独立性未通过'
    return '动量检查未通过'


def linear(terms):
    """Render exact rational coefficients, including negatives and duplicates."""
    parts = []
    for coefficient, symbol in terms:
        value = Fraction(str(coefficient))
        if not value:
            continue
        magnitude = abs(value)
        number = (str(magnitude.numerator) if magnitude.denominator == 1 else
                  rf'\frac{{{magnitude.numerator}}}{{{magnitude.denominator}}}')
        term = symbol if magnitude == 1 else number + ' ' + symbol
        parts.append(('-' if value < 0 else '+' if parts else '') + term)
    return ' '.join(parts) or '0'


def subset_symbol(subset):
    return 's_{' + ','.join(map(str, subset)) + '}'


def dependence_equation(certificate):
    left = linear(zip(certificate['multipliers'],
                      map(subset_symbol, certificate['variables'])))
    mass = certificate['rhs_mass']
    right = linear([(mass['parent_coefficient'], 'M^2')] + [
        (value, f'm_{{{i}}}^2') for i, value in enumerate(mass['particle_coefficients'], 1)])
    return left + ' = ' + right


def table(headers, rows):
    head = ''.join('<th>' + escape(str(h)) + '</th>' for h in headers)
    body = ''.join('<tr>' + ''.join('<td>' + escape(str(v)) + '</td>' for v in row)
                   + '</tr>' for row in rows)
    return '<table class="psg-table"><thead><tr>' + head + '</tr></thead><tbody>' + body + '</tbody></table>'


class AuditBrowser:
    def __init__(self, root):
        import ipywidgets as w
        from IPython.display import display, HTML, Markdown, Math, clear_output
        self.w, self.display, self.HTML, self.Markdown, self.Math = w, display, HTML, Markdown, Math
        self.clear = clear_output
        self.directory = Path(root) / '文档' / '验证与测试' / '结果图表' / '四体图枚举'
        self.catalog_directory = Path(root) / '文档' / '验证与测试' / '结果图表' / '独立标记'
        self.audit = None
        self.catalog = None
        self.dataset = None
        self.source_kind = None
        self.busy = False
        self.updating = False
        self.loop = w.Dropdown(options=[('树图（l=0）', 0), ('单圈（l=1）', 1),
                                       ('两圈（l=2）', 2), ('三圈（l=3）', 3)], description='图类型')
        self.source = w.Dropdown(description='结果文件', layout=w.Layout(width='95%'))
        self.mode = w.Dropdown(options=[('固定粒子编号', CountMode.FIXED_LABELS.value),
                                       ('整体粒子换名', CountMode.PARTICLE_RELABELING.value)],
                              value=CountMode.FIXED_LABELS.value, description='标记口径')
        self.filter = w.Dropdown(options=['有效拓扑', '独立性未通过', '动量检查未通过',
                                         '待完成／运行失败', '全部拓扑'], description='筛选')
        self.graph = w.Dropdown(description='拓扑', layout=w.Layout(width='95%'))
        self.label = w.Dropdown(description='粒子标记', layout=w.Layout(width='95%'))
        self.refresh = w.Button(description='刷新已保存结果', icon='refresh')
        self.run = w.Button(description='重新计算当前圈数', button_style='warning')
        self.save = w.Button(description='另存当前结果', icon='save', disabled=True)
        self.notice = w.HTML()
        self.summary, self.detail = w.Output(), w.Output()
        self.controls = [self.loop, self.source, self.mode, self.filter, self.graph, self.label,
                         self.refresh, self.run, self.save]
        self.loop.observe(lambda change: self._files(), names='value')
        self.mode.observe(self._change_mode, names='value')
        self.source.observe(lambda change: self._load(), names='value')
        self.filter.observe(lambda change: self._graphs(), names='value')
        self.graph.observe(lambda change: self._labels(), names='value')
        self.label.observe(lambda change: self._render(), names='value')
        self.refresh.on_click(lambda button: self._files())
        self.run.on_click(self._run)
        self.save.on_click(self._save)
        self.view = w.VBox([
            w.HTML('<h2>四体图检查面板</h2><p>① 选圈数 → ② 选拓扑 → ③ 切换标记。'
                   '默认读取已有结果；只有点击“重新计算”才开始枚举。</p>'
                   '<style>.psg-table{border-collapse:collapse;margin:12px 0;width:100%}'
                   '.psg-table td,.psg-table th{border:1px solid #ddd;padding:7px;text-align:left}'
                   '.psg-table th{background:#f1f5f9;color:#17212b}</style>'),
            self.loop, self.source, self.mode, w.HBox([self.refresh, self.run, self.save]), self.notice,
            self.summary, self.filter, self.graph, self.label, self.detail])

    def show(self):
        self.display(self.view)
        self._files()
        return self

    def _message(self, message):
        self.notice.value = '<p>' + escape(message) + '</p>'

    def _catalog_view(self):
        """Use the new card view when the selected file has concrete records.

        Tiny legacy fixtures in old tests may contain only summary-shaped
        labeling entries.  They remain on the old diagnostic renderer instead
        of being presented as fabricated physical cards.
        """

        if self.catalog is None:
            return False
        if not any(item.mode.value == self.mode.value for item in self.catalog.catalogs):
            return False
        if self.source_kind == CATALOG_ARTIFACT:
            try:
                selected = CountMode(self.mode.value)
            except ValueError:
                return False
            return any(item.mode is selected for item in self.catalog.catalogs)
        return any(
            catalog.identities or any(witness.record is not None for witness in catalog.rejected_witnesses)
            for catalog in self.catalog.catalogs
        )

    def _files(self):
        if self.busy:
            return
        clear_browser_source_cache()
        paths = list(self.directory.glob(f'g08_四体_l{self.loop.value}_*.json'))
        paths.extend(self.catalog_directory.glob(f'label_catalog_四体_l{self.loop.value}_*.json'))
        paths = sorted(paths, key=lambda p: p.stat().st_mtime_ns, reverse=True)
        old = self.source.value
        values = [str(p) for p in paths]
        self.updating = True
        self.source.options = [
            (f"{'旧审计' if p.parent == self.directory else '独立标记目录'}：{p.stem}", str(p))
            for p in paths
        ]
        self.source.value = old if old in values else values[0] if values else None
        self.updating = False
        self._load()

    def _change_mode(self, change):
        if self.updating or self.busy:
            return
        if self.source.value:
            self._load()
            return
        # Changing the view of an unsaved computation must not erase it.
        if self.audit is not None:
            try:
                converted = catalog_from_audit(self.audit, mode=self.mode.value)
            except Exception as exc:
                self.updating = True
                self.mode.value = change['old']
                self.updating = False
                self._message(f'口径切换失败，保留当前结果：{exc}')
                return
            self.catalog = converted
        self._summary()
        self._graphs()
        self._message('已切换显示口径；未另存的计算仍保留。缺失口径不会自动补算。')

    def _load(self):
        if self.updating or self.busy:
            return
        previous = (self.audit, self.catalog, self.dataset, self.source_kind)
        try:
            if not self.source.value:
                self.audit = None
                self.catalog = None
                self.dataset = None
                self.source_kind = None
                self.save.disabled = True
                self._message('这一圈还没有保存结果，可选择其他圈数。')
            else:
                path = Path(self.source.value)
                source = load_browser_source(path, mode=self.mode.value)
                if source.loop_count is not None and source.loop_count != self.loop.value:
                    raise ValueError('文件记录的 n/l 与选择不符')
                self.dataset = source
                self.source_kind = source.source_kind
                self.audit = source.audit
                self.catalog = source.catalog
                cache_note = '（使用缓存）' if source.loaded_from_cache else '（首次读取并转换）'
                self._message(f'已加载：{path.name}{cache_note}。当前口径：{self.mode.value}。')
                self.save.disabled = False
        except Exception as exc:
            self.audit, self.catalog, self.dataset, self.source_kind = previous
            self._message(f'读取失败：{exc}。请确认文件写入已结束，再点刷新。')
        self._summary()
        self._graphs()

    def _summary(self):
        with self.summary:
            self.clear(wait=True)
            if self.audit is None and self.catalog is None:
                return
            if not self._catalog_view() and self.audit is None:
                self.display(self.Markdown(f'当前文件没有 `{self.mode.value}` 口径的目录；请切换文件或先按此口径重新计算。'))
                return
            if self._catalog_view():
                catalogs = tuple(c for c in self.catalog.catalogs if c.mode.value == self.mode.value) if self.catalog else ()
                valid_topologies = sum(item.identity_count > 0 for item in catalogs)
                identity_count = sum(item.identity_count for item in catalogs)
                variable_keys = {link.variable_key for item in catalogs for link in item.links}
                dn_keys = {link.dn_label for item in catalogs for link in item.links}
                complete = bool(self.catalog and self.catalog.enumeration_complete)
                self.display(self.HTML(table(['项目', '数量／状态'], [
                    ['数据来源', '新标记目录' if self.source_kind == CATALOG_ARTIFACT else '旧审计转换目录'],
                    ['标记口径', self.mode.value],
                    ['拓扑目录数', len(catalogs)],
                    ['存在有效方案的拓扑数', valid_topologies],
                    ['独特有效标记数', identity_count],
                    ['变量组合类数', len(variable_keys)],
                    ['DN 类数', len(dn_keys)],
                    ['全体完整性', '声明范围的拓扑与标记均已找全' if complete else '拓扑范围或标记尚未证明完整'],
                    ['导入诊断', sum(len(c.issues) for c in catalogs) + len(self.catalog.issues)],
                ])))
                self.display(self.Markdown('“存在有效方案”与“已经找全”分开显示；未完成目录不能解释为无有效图。'))
                return
            reports = self.audit.get('labeling_reports', [])
            counts = self.audit.get('counts', {})
            self.display(self.HTML(table(['项目', '数量／状态'], [
                ['本次搜索', STATES.get(self.audit.get('status'), '状态未知')],
                ['结构候选', counts.get('topologies', '未记录')],
                ['至少有一种动量合格标记', sum(r.get('counts', {}).get('momentum_valid', 0) > 0 for r in reports)],
                ['至少有一种有效独立标记', sum(bool(r.get('valid_results')) for r in reports)],
                ['变量组合类', counts.get('variable_combination_classes', '未记录')],
                ['DN 类', counts.get('dn_classes', '未记录')]])))
            runtime = self.audit.get('runtime', {})
            self.display(self.Markdown('搜索完成不等于文献逐图验收通过。未完成搜索中的数目是当前已找到的数量。'))
            self.display(self.HTML('<small>运行时间：' + escape(str(self.audit.get('started_at', '未记录')))
                                   + '；代码版本：' + escape(str(runtime.get('git_commit', '未记录'))) + '</small>'))

    def _graphs(self):
        if self.updating or self.busy:
            return
        if self._catalog_view():
            catalogs = self.catalog.catalogs if self.catalog else ()
            options = []
            self._catalog_indices = []
            for index, catalog in enumerate(catalogs):
                if catalog.mode.value != self.mode.value:
                    continue
                status = _catalog_status(catalog)
                witness_stages = {witness.stage for witness in catalog.rejected_witnesses}
                include = (
                    self.filter.value == '全部拓扑'
                    or (self.filter.value == '有效拓扑' and catalog.identity_count > 0)
                    or (self.filter.value == '独立性未通过' and 'dependent' in witness_stages)
                    or (self.filter.value == '动量检查未通过' and bool(witness_stages & {'momentum_rejected', 'field_rejected', 'structure_rejected'}))
                    or (self.filter.value == '待完成／运行失败' and not catalog.enumeration_complete)
                )
                if include:
                    options.append((f'拓扑 {index + 1} · {status}', index))
                    self._catalog_indices.append(index)
            old = self.graph.value
            self.updating = True
            self.graph.options = options
            values = [value for _, value in options]
            self.graph.value = old if old in values else values[0] if values else None
            self.updating = False
            self._labels()
            return
        reports = self.audit.get('labeling_reports', []) if self.audit else []
        options = [(f'拓扑 {i + 1} · {stage(r)}', i) for i, r in enumerate(reports)
                   if self.filter.value == '全部拓扑' or stage(r) == self.filter.value]
        old = self.graph.value
        self.updating = True
        self.graph.options = options
        values = [v for _, v in options]
        self.graph.value = old if old in values else values[0] if values else None
        self.updating = False
        self._labels()

    def _labels(self):
        if self.updating or self.busy:
            return
        self.records = []
        if self._catalog_view() and self.graph.value is not None:
            catalog = self.catalog.catalogs[self.graph.value]
            for index, identity in enumerate(catalog.identities, 1):
                variable_labels = ', '.join(
                    's' + ''.join(str(item) for item in variable)
                    for variable in identity.canonical_record.momentum.variables
                )
                normalization = NormalizationResult('accepted', record=identity.canonical_record)
                self.records.append((f'有效标记类 {index}：{variable_labels}', build_physics_card(normalization, identity=identity)))
            for witness in catalog.rejected_witnesses:
                if witness.stage != 'accepted':
                    title = '失败代表：双线不变质量之间存在依赖关系' if witness.stage == 'dependent' else '失败代表：' + '；'.join(issue.message for issue in witness.issues)
                    self.records.append((title, build_physics_card(witness)))
            self.updating = True
            self.label.options = [(title, index) for index, (title, _) in enumerate(self.records)]
            self.label.value = 0 if self.records else None
            self.updating = False
            self._render()
            return
        if self.audit is not None and self.graph.value is not None:
            report = self.audit['labeling_reports'][self.graph.value]
            for index, result in enumerate(report.get('valid_results', []), 1):
                names = ', '.join((result.get('independence') or {}).get('labels', []))
                self.records.append((f'有效标记类 {index}：{names}', result))
            for code, result in report.get('failure_witnesses', {}).items():
                self.records.append(('失败代表：' + reason(code), result))
        self.updating = True
        self.label.options = [(title, i) for i, (title, _) in enumerate(self.records)]
        self.label.value = 0 if self.records else None
        self.updating = False
        self._render()

    def _render(self):
        if self.updating:
            return
        if self._catalog_view() and self.graph.value is not None:
            self._render_catalog()
            return
        with self.detail:
            self.clear(wait=True)
            if self.audit is None or self.graph.value is None:
                self.display(self.Markdown('当前筛选下没有拓扑。'))
                return
            report = self.audit['labeling_reports'][self.graph.value]
            topo = report['topology']
            self.display(self.Markdown(f'### 拓扑 {self.graph.value + 1}：{stage(report)}'))
            self.display(self.Markdown('编号仅用于当前结果文件内定位，不是文献图号。'))
            self.display(self.HTML(table(['内部边', '方向', '线型', '条数'], [
                [i + 1, f"{e['source']} → {e['target']}",
                 '双线' if e['edge_type'] == 'double' else '单线', e['multiplicity']]
                for i, e in enumerate(topo['edges'])])))
            self.display(self.HTML(table(['顶点', '末态外线数'], [
                [s['vertex'], s['count']] for s in topo['final_slots']])))
            self.display(self.Markdown(f"初态总动量 $P$ 进入顶点 {topo['initial_vertex']}。"))
            failures = report.get('failure_counts', {})
            if failures:
                self.display(self.HTML(table(['标记被拒绝的原因', '累计次数'],
                    [[reason(code), count] for code, count in failures.items()])))
                self.display(self.Markdown('这里统计的是标记失败次数；同一标记可能触发多个原因，不能相加当作图数。'))
            if self.label.value is None:
                self.display(self.Markdown('此记录没有可展示的完整标记证据。旧审计可能仅保存原因统计；没有证据不等于没有失败。'))
                self._raw(report)
                return
            result = self.records[self.label.value][1]
            labels = result['labeling']
            self.display(self.Markdown('### 粒子标记'))
            self.display(self.HTML(table(['位置', '粒子'], [
                [f"顶点 {p['vertex']} 的末态槽 {p['slot'] + 1}", p['particle_id']]
                for p in labels.get('final_particles', [])] + [
                [f"内部边 {p['edge_index'] + 1}，第 {p['occurrence'] + 1} 条单线", p['particle_id']]
                for p in labels.get('single_lines', [])])))
            momentum = result.get('momentum')
            if momentum:
                self._physics(topo, labels, momentum, result.get('independence'))
            else:
                self.display(self.Markdown('该标记未得到可展示的双线动量证书。'))
            if result.get('failure_codes'):
                self.display(self.Markdown('所选标记未通过：' + '；'.join(map(reason, result['failure_codes']))))
            self._raw({'topology': topo, 'selected_labeling': result})

    def _render_catalog(self):
        with self.detail:
            self.clear(wait=True)
            if self.catalog is None or self.graph.value is None:
                self.display(self.Markdown('当前筛选下没有拓扑。'))
                return
            catalog = self.catalog.catalogs[self.graph.value]
            self.display(self.Markdown(
                f'### 拓扑 {self.graph.value + 1}：{_catalog_status(catalog)}\n\n'
                f'当前标记口径：`{catalog.mode.value}`；目录中的独特标记数：`{catalog.identity_count}`。'
            ))
            if catalog.issues:
                self.display(self.HTML(table(['导入问题', '说明'], [(i.code, i.message) for i in catalog.issues])))
            self.display(self.Markdown(
                '目录完整性和物理存在性分开记录：'
                + ('本目录已找全。' if catalog.enumeration_complete else '本目录尚未证明找全。')
            ))
            if self.label.value is None:
                self.display(self.Markdown('当前拓扑没有可展示的完整记录；已有失败代表也可能不是失败全集。'))
                return
            selected = self.records[self.label.value][1]
            self.display(self.Markdown(render_card_markdown(selected)))

    def _physics(self, topo, labels, momentum, independence):
        self.display(self.Markdown('### 双线动量与不变质量'))
        symbols = {}
        for item in momentum['double_momenta']:
            key = item['edge_index'], item['occurrence']
            q = f'q_{{{key[0] + 1},{key[1] + 1}}}'
            symbols[key] = q
            coefficients = item['coefficients']
            vector = linear((c, f'p_{{{i}}}') for i, c in enumerate(coefficients, 1))
            proper = all(c in (0, 1) for c in coefficients) and 2 <= sum(coefficients) < topo['n']
            equation = q + ' = ' + vector
            if proper:
                subset = [i for i, c in enumerate(coefficients, 1) if c == 1]
                equation += r' ,\qquad ' + q + '^2 = ' + subset_symbol(subset)
            self.display(self.Math(equation))
        self.display(self.Markdown(r'其中 $s_A=(\sum_{i\in A}p_i)^2$。非合法复合粒子动量保留原系数，不强行写成子集质量。'))
        self.display(self.Markdown('### 逐顶点守恒方程'))
        singles = {(p['edge_index'], p['occurrence']): f"p_{{{p['particle_id']}}}"
                   for p in labels.get('single_lines', [])}
        for vertex in topo['vertices']:
            incoming = ['P'] if vertex == topo['initial_vertex'] else []
            outgoing = [f"p_{{{p['particle_id']}}}" for p in labels.get('final_particles', []) if p['vertex'] == vertex]
            for i, edge in enumerate(topo['edges']):
                for occurrence in range(edge['multiplicity']):
                    symbol = (symbols if edge['edge_type'] == 'double' else singles).get((i, occurrence))
                    symbol = symbol or f'x_{{{i + 1},{occurrence + 1}}}'
                    if edge['target'] == vertex:
                        incoming.append(symbol)
                    if edge['source'] == vertex:
                        outgoing.append(symbol)
            self.display(self.Math(rf'\text{{顶点 {vertex}: }}' + (' + '.join(incoming) or '0')
                                   + ' = ' + (' + '.join(outgoing) or '0')))
        self.display(self.HTML(table(['顶点', '回代残差（按粒子系数）'],
            [[v['vertex'], str(v['residual'])] for v in momentum.get('vertex_residuals', [])])))
        self.display(self.Markdown('守恒证书：' + ('已验证' if momentum.get('verified') else '未验证')
                                  + '。守恒成立不代表双线质量一定独立。'))
        self.display(self.Markdown('### 不变质量独立性'))
        if not independence:
            self.display(self.Markdown('尚未进行独立性检查：先前检查未通过或证书缺失。'))
            return
        d = len(independence['variables'])
        self.display(self.Markdown(f"变量数 {d}，精确秩 {independence['rank']}："
                                  + ('相互独立。' if independence['independent'] else '存在依赖关系。')))
        certificate = independence.get('certificate')
        if certificate:
            self.display(self.Math(dependence_equation(certificate)))
            self.display(self.Markdown('上式来自保存的精确依赖证书，回代状态：'
                                      + ('已验证。' if certificate.get('verified') else '未验证。')))

    def _raw(self, data):
        content = '<pre style="max-height:260px;overflow:auto">' + escape(json.dumps(data, ensure_ascii=False, indent=2)) + '</pre>'
        accordion = self.w.Accordion(children=[self.w.HTML(content)], selected_index=None)
        accordion.set_title(0, '原始记录（可选，排查问题时再展开）')
        self.display(accordion)

    def _set_busy(self, busy):
        self.busy = busy
        for control in self.controls:
            control.disabled = busy
        if not busy:
            self.save.disabled = self.audit is None and self.catalog is None

    def _run(self, button):
        if self.busy:
            return
        loop = self.loop.value
        self._set_busy(True)
        self._message(f'正在重新计算 l={loop}，请等待。完成后自动更新面板；本次暂不写入文件。')
        try:
            fresh_result = run_label_catalog(4, loop, mode=self.mode.value, use_symmetry=True)
            fresh_audit = fresh_result.legacy_audit.to_dict()
            fresh_catalog = fresh_result.label_catalog
            # Install the result only once the entire conversion succeeds.
            self.audit, self.catalog = fresh_audit, fresh_catalog
            self.dataset = None
            self.source_kind = 'fresh_run'
            self.updating = True
            self.source.options = [('本次计算（尚未另存）', None)]
            self.source.value = None
            self.updating = False
            self._message('本次计算已返回，面板已切换到新结果。请查看搜索完整性；需要保留时点击另存。')
        except KeyboardInterrupt:
            self._message('计算已中断；保留面板先前的数据。')
        except Exception as exc:
            self._message(f'计算失败，保留先前数据：{exc}')
        finally:
            self._set_busy(False)
        self._summary()
        self._graphs()

    def _save(self, button):
        if self.busy or (self.audit is None and self.catalog is None):
            return
        try:
            loop = self.audit.get('topology_report', {}).get('loop_count') if self.audit else None
            if loop is None and self.catalog and self.catalog.catalogs:
                loop = self.catalog.catalogs[0].search_evidence.loop_count
            if loop is None:
                loop = self.loop.value
            if self.source_kind in {CATALOG_ARTIFACT, 'fresh_run'} and self.catalog is not None:
                name = f'label_catalog_四体_l{loop}_交互检查_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}.json'
                self.catalog_directory.mkdir(parents=True, exist_ok=True)
                path = self.catalog_directory / name
                write_label_catalog(self.catalog, path)
            else:
                name = f'g08_四体_l{loop}_交互检查_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}.json'
                self.directory.mkdir(parents=True, exist_ok=True)
                path = self.directory / name
                # Exclusive creation: historical files can never be overwritten.
                with path.open('x', encoding='utf-8') as stream:
                    json.dump(self.audit, stream, ensure_ascii=False)
            self.updating = True
            self.source.options = [(name, str(path))]
            self.source.value = str(path)
            self.updating = False
            self._message('已另存为：' + name + '。刷新可重新列出所有历史文件。')
        except Exception as exc:
            self._message(f'另存失败：{exc}。内存中的结果仍可查看。')


__all__ = [
    'AuditBrowser',
    'BrowserSource',
    'load_browser_source',
    'clear_browser_source_cache',
    'reason',
    'stage',
    'linear',
    'subset_symbol',
    'dependence_equation',
    'table',
]
