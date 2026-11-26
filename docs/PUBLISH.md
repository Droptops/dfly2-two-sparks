# Publishing this repository

Create an empty repository under the account you want to own it. This project already contains a README, license, notices, and ignore rules, so do not initialize additional versions on GitHub.

The prepared local repository uses `main`. Once the new repository URL exists, connect it and push:

```sh
git remote add origin https://github.com/YOUR-ACCOUNT/YOUR-REPO.git
git push -u origin main
```

Use your normal GitHub authentication. Credentials and local cluster configuration do not belong in this repository.

Suggested description:

> Experimental native DFly2 speculative decoding for DeepSeek on two GB10 Sparks, with reproducible setup and measured comparisons.

Suggested topics: `dgx-spark`, `gb10`, `deepseek`, `speculative-decoding`, `cuda`, `zig`, `tensorfold`.

The CPU/evidence/source checks run automatically on pushes and pull requests. GPU tests are explicit hardware checks. Keep the experimental label and the DSpark comparison visible in an initial release announcement; the measured DFly2 port was slower than DSpark on the published workload.

The sanitized files in `evidence/` are the publication data. Raw local experiment archives, model weights, host configuration, build caches, and remote-service state are outside this release.
