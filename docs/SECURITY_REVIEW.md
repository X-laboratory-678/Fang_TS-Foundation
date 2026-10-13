# Security and rights review for public release

Updated 2026-10-13 (Asia/Shanghai); includes pre- and post-publication checks.

| Check | Status | Evidence / limit |
|---|---|---|
| Secrets and private keys in new candidate text | PASS for known patterns | Candidate-file scan found no GitHub-token, AWS-key, private-key, or credential-assignment pattern. Pattern scanning cannot establish absence of all secrets. |
| Absolute personal paths in candidate code/config | PASS for selected files | Local model paths in four YAMLs and one generator were replaced with placeholders/configuration; the MAML debug path is configurable. Original SHA-256 hashes and edits are documented. |
| Weights, raw private data, cache, oversized files | PASS for new files | No checkpoints/raw datasets copied; new `research_code/` is under 1 MB. `.gitignore` excludes common private/runtime outputs. |
| Local source credentials | PASS for exclusion | The real experiment tree's `.env.local` was not copied. Its contents were not put in this report. |
| Upstream attribution | PARTIAL | LLaMA-Factory and Apache-2.0 notices are retained. Exact upstream revision of the local modified tree cannot be recovered from its empty `.git`. |
| Dataset/model redistribution rights | PARTIAL | Raw datasets and weights for the paper experiments are excluded. Exact provider terms and version identity remain unverified. |
| Existing remote history JSON permission | PASS based on author confirmation | The author confirmed on 2026-10-13 that public release is authorized for the unrelated data-center task JSON already present in public commit `12cd209` (`sft_train.json` and `sft_test.json`, about 25.9 MB combined). The files were not modified or newly copied. A scan found no email, Chinese mobile, IPv4, credential-token, private-key, URL-userinfo, personal absolute-path, or sensitive-field-name patterns; this is not a complete legal/privacy determination. No public history was rewritten. |
| Existing-history credential patterns | PASS for known patterns | All 14 reachable commits were scanned for common GitHub/AWS/private-key markers and long credential assignments; no matches. This does not address rights to the existing JSON. |
| Publication and access | PASS for branch publication; PARTIAL for direct Git credentials | Published through the authenticated `X-laboratory-678` browser session to `patterns-real-code-prep-20261012`. The GitHub API connector remains `King166969` with `push: false`; terminal Git has no HTTPS/SSH credential. No force push, history rewrite, PR, or merge occurred; `main` is unchanged. |

**Post-publication status (2026-10-13):** The author-confirmed permission resolved the prior hold for the two inherited JSON files. An anonymous clone of the public release branch succeeded with credential helpers and authorization headers disabled; all 52 release files were present and their content hashes matched the prepared snapshot. Third-party dataset/model rights remain unverified, and those assets are excluded. The default branch was not changed.
