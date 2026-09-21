# JOSS readiness checklist

This file records preparation status; it is not a submission or an acceptance claim.

## Present in this repository

- public source tree for the four-body and five-body demos;
- installable package metadata for each demo;
- unit tests and a GitHub Actions workflow;
- an explicit MIT license;
- README files describing installation, examples, scope, and limitations;
- contribution and support guidance;
- a documented separation between software tests and research evidence.

The repository also keeps the expensive five-body L5 counter outside the
default `unittest discover` pattern; it is an internal long-running check,
not a quick installation smoke test.

## Still required before submission

- maintain the repository publicly over time and continue development across releases;
- document concrete research use and the software’s scholarly contribution;
- finalize author names, affiliations, and contribution statements;
- add `paper.md` and the associated bibliography/figures;
- decide which public release and generated artifacts are the citable version;
- review third-party and source-material licensing before adding any reference material;
- respond to external issues and feedback through the public repository.

The current repository should therefore be treated as an initial public maintenance point for the demos, not as a ready-to-submit JOSS package.
