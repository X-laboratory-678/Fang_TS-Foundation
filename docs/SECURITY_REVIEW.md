# Pre-publication security and rights review

Updated 2026-10-13 against the local release branch, before any push.

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
| Push channel | PARTIAL | The browser session is `X-laboratory-678`; GitHub showed the ChatGPT Codex Connector installation updated with this repository selected. A connector metadata call failed at transport level and terminal Git has no HTTPS credential, so a working push route still requires verification. |

**Pre-push status:** The author-confirmed permission and pattern scan resolved the prior hold for the two inherited JSON files. Third-party dataset/model rights remain unverified but those assets are excluded. The research-code package remains local pending a successful push-channel check.
