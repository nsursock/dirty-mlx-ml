# General

## Writing Commit Messages

Imperative Mood in Subject: Write the summary line in the imperative present tense (e.g., Add user authentication or Fix memory leak in buffer, rather than Added or Fixes). This aligns with standard Git generator convention (If applied, this commit will <subject>).

Subject Line Formatting: Limit the first line to 50–72 characters, capitalize the first letter, and do not end the line with a period.

Structured Body: Separate the subject line from the body with a blank line. Use the body to explain what changed and why, rather than how (the diff shows how).

Reference Issue Tracking: Link relevant issue numbers or pull request references in the commit body (e.g., Closes #142 or Ref #88).

## Libraries / Development

Don't install SB3 or Gymnasium. Use only MLX.