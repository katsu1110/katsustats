---
name: 'make-skill-template'
description: 'Create new GitHub Copilot agent skills for this repository. Use when asked to create a skill, scaffold a new skill, add a reusable agent workflow, or define SKILL.md frontmatter, structure, and optional bundled resources.'
license: 'Apache-2.0'
---

# Make Skill Template

Create a new agent skill for this repository.

Use this skill when a user wants a reusable Copilot workflow packaged as a skill folder with a `SKILL.md` file and, if needed, bundled supporting files.

## When to Use This Skill

Use this skill when:
- The user asks to create or scaffold a new skill
- The user wants a reusable workflow for GitHub Copilot
- A task would benefit from a dedicated slash-command style skill
- The user needs help structuring `SKILL.md` frontmatter and instructions

## Skill Structure Rules

Create project skills under `.github/skills/<skill-name>/`.

Each skill must have:
- A lowercase, hyphenated folder name
- A `SKILL.md` file
- A `name` field that matches the folder name exactly
- A `description` field that explains both WHAT the skill does and WHEN it should be used

Optional subfolders may be added when useful:
- `references/` for longer documentation
- `templates/` for starter content the agent can adapt
- `assets/` for static files used as-is
- `scripts/` for helper automation

## Step-by-Step Workflow

### 1. Define the skill scope

Identify:
- The single main purpose of the skill
- The user triggers and keywords that should activate it
- Any repository-specific constraints or conventions
- Whether supporting files are needed

Prefer one focused purpose per skill.

### 2. Choose the skill name

The skill name must:
- be lowercase
- use hyphens instead of spaces
- be short, descriptive, and action-oriented
- match the folder name exactly

Good examples:
- `publish`
- `make-skill-template`
- `generate-report`

Avoid vague names like:
- `helper`
- `tools`
- `misc`

### 3. Create `SKILL.md`

Use this structure:

```markdown
---
name: 'skill-name'
description: 'What this skill does. Use when the user asks for X, Y, or Z, or when the task involves A and B.'
license: 'Apache-2.0'
---

# Skill Title

One short paragraph describing the skill.

## When to Use This Skill

- Trigger 1
- Trigger 2
- Trigger 3

## Requirements

- Important constraint 1
- Important constraint 2

## Step-by-Step Workflow

### 1. First step

What to do.

### 2. Second step

What to do.

## Gotchas

- Common mistake 1
- Common mistake 2

## References

- `references/example.md`
```

### 4. Write a strong description

The `description` is the main discovery signal.

It should include:
1. What the skill does
2. When to use it
3. Likely trigger words the user may say

Good description pattern:
- `Create release notes for this repository. Use when asked to summarize commits, prepare changelog text, or draft GitHub release notes.`

Weak description pattern:
- `Release helper.`

## Requirements

When creating a new skill:
- Keep instructions specific and actionable
- Prefer repository-specific guidance over generic advice
- Keep the skill focused on one repeatable workflow
- Keep changes minimal when scaffolding files
- Reference bundled files with relative paths if they exist
- Do not add optional directories unless they provide real value

## Gotchas

- Do not let the `name` differ from the folder name
- Do not use a vague `description`; weak discovery makes the skill hard to trigger
- Do not duplicate long reference content in `SKILL.md`; place it in `references/` instead
- Do not create oversized or unnecessary bundled assets
- Do not make the skill task-specific if it should be reusable

## Validation Checklist

Before finishing, confirm that:
- Folder name is lowercase and hyphenated
- `SKILL.md` exists in the skill folder
- `name` matches the folder name exactly
- `description` clearly explains WHAT and WHEN
- Body instructions are concise and reusable
- Any referenced bundled files actually exist

## Example Output

For a skill named `draft-changelog`, the result should look like:

```text
.github/skills/draft-changelog/
└── SKILL.md
```

With frontmatter like:

```yaml
name: 'draft-changelog'
description: 'Draft changelog entries for this repository. Use when asked to summarize recent changes, prepare release notes, or group commits into user-facing sections.'
```

## References

- Agent skills in this repository live under `.github/skills/`
- Use existing skills in this repository as style references when relevant
