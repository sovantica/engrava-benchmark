// Commitlint config — enforces Conventional Commits 1.0.0 + the project scope allowlist.
// Reference: https://commitlint.js.org/
// Used by .github/workflows/commitlint.yml to validate PR titles + commits.

module.exports = {
  extends: ["@commitlint/config-conventional"],

  // Skip non-Conventional, git-generated commits (merge/revert).
  ignores: [
    (message) => message.startsWith("Merge ") || message.startsWith("Revert "),
    (message) => /Signed-off-by: dependabot\[bot\]/.test(message),
  ],

  rules: {
    "type-enum": [
      2,
      "always",
      ["feat", "fix", "perf", "docs", "style", "refactor", "test", "build", "ci", "chore", "revert"],
    ],

    "scope-enum": [
      2,
      "always",
      ["adapters", "runners", "results", "schema", "scripts", "docs", "ci", "build", "deps", "deps-dev", "release"],
    ],

    "scope-empty": [1, "never"],
    "header-max-length": [2, "always", 100],
    "subject-full-stop": [2, "never", "."],
    "body-leading-blank": [2, "always"],
    "footer-leading-blank": [2, "always"],
    "type-empty": [2, "never"],
    "subject-empty": [2, "never"],
  },
};
