# Git Submodule Management Guide

This guide explains how to work with the prompts submodule in the AI-NetGuard-Core development repository.

## Overview

The `prompts/` directory is a Git submodule that links to the [AI-NetGuard](https://github.com/mdastpak/AI-NetGuard) repository. This setup provides read-only access to the JSON prompt files while keeping the development code separate.

## Important Notes

- **Read-Only Access**: You cannot modify files in the `prompts/` directory directly.
- **Separate Repository**: All changes to prompts must be made in the main AI-NetGuard repository.
- **Version Control**: The submodule tracks a specific commit from the main repository.

## Initial Setup

When cloning the repository for the first time:

```bash
git clone https://github.com/mdastpak/AI-NetGuard-Core.git
cd AI-NetGuard-Core
git submodule init
git submodule update
```

Or use the provided setup script:

```bash
./setup.sh
```

## Updating Prompts

To update the prompts to the latest version from the main repository:

```bash
git submodule update --remote
```

This will:
1. Fetch the latest changes from the AI-NetGuard repository
2. Update the submodule to point to the latest commit on the main branch
3. Update your working directory

## Checking Submodule Status

To check the status of submodules:

```bash
git submodule status
```

This shows:
- The current commit hash of the submodule
- Whether it's up to date
- Any uncommitted changes

## Making Changes to Prompts

If you need to modify prompt files:

1. Go to the main [AI-NetGuard repository](https://github.com/mdastpak/AI-NetGuard)
2. Make your changes there
3. Commit and push the changes
4. Return to AI-NetGuard-Core and update the submodule:
   ```bash
   git submodule update --remote
   git add prompts
   git commit -m "Update prompts to latest version"
   git push
   ```

## Troubleshooting

### Submodule Not Initialized

If you see an empty `prompts/` directory:

```bash
git submodule init
git submodule update
```

### Permission Issues

Ensure you have read access to the AI-NetGuard repository.

### Conflicting Changes

If there are conflicts during updates, the submodule will show as modified. To resolve:

```bash
git submodule update --init --recursive
```

### Removing Submodule (Advanced)

To remove the submodule (not recommended):

```bash
git submodule deinit prompts
git rm prompts
git commit -m "Remove prompts submodule"
```

## Best Practices

1. Always update submodules before starting development
2. Commit submodule updates separately from code changes
3. Test your code after updating prompts
4. Keep the submodule on the main branch (not a specific commit unless necessary)

## Contact

For issues with prompts or submodule setup, refer to the main repository's documentation.