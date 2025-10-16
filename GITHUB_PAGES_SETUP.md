# GitHub Pages Setup Guide for SOLWEIG-GPU

This guide explains how to set up and deploy the SOLWEIG-GPU documentation to GitHub Pages.

## Overview

The documentation is built using **MkDocs** with the **Material for MkDocs** theme. It is automatically deployed to GitHub Pages using GitHub Actions whenever changes are pushed to the `main` branch.

## Prerequisites

Before setting up GitHub Pages, ensure you have:

1. A GitHub repository for SOLWEIG-GPU
2. Admin access to the repository
3. The documentation files in the `docs/` directory
4. The `mkdocs.yml` configuration file in the repository root

## Setup Steps

### Step 1: Enable GitHub Pages

1. Go to your repository on GitHub
2. Click on **Settings** (in the repository menu)
3. Scroll down to the **Pages** section in the left sidebar
4. Under **Source**, select **Deploy from a branch**
5. Select the `gh-pages` branch and `/ (root)` folder
6. Click **Save**

### Step 2: Configure Repository Permissions

The GitHub Actions workflow needs permission to deploy to GitHub Pages:

1. In **Settings**, go to **Actions** → **General**
2. Scroll down to **Workflow permissions**
3. Select **Read and write permissions**
4. Check **Allow GitHub Actions to create and approve pull requests**
5. Click **Save**

### Step 3: Push the Documentation Files

Make sure all the documentation files are committed and pushed to your repository:

```bash
git add docs/ mkdocs.yml .github/workflows/docs.yml
git commit -m "Add documentation and GitHub Pages deployment"
git push origin main
```

### Step 4: Verify Deployment

1. Go to the **Actions** tab in your repository
2. You should see a workflow run called "Deploy Documentation"
3. Wait for the workflow to complete (it should take 1-2 minutes)
4. Once complete, your documentation will be available at:
   ```
   https://<username>.github.io/<repository-name>/
   ```

For the SOLWEIG-GPU repository, this would be:
```
https://nvnsudharsan.github.io/SOLWEIG-GPU/
```

## Local Development

To preview the documentation locally before pushing:

### Install MkDocs

```bash
pip install mkdocs-material mkdocstrings[python] pymdown-extensions
```

### Preview Locally

```bash
cd /path/to/SOLWEIG-GPU
mkdocs serve
```

This will start a local server at `http://127.0.0.1:8000/` where you can preview the documentation.

### Build Locally

To build the documentation without serving it:

```bash
mkdocs build
```

This creates a `site/` directory with the static HTML files.

## Customization

### Updating the Theme

The documentation uses the Material for MkDocs theme. You can customize it by editing `mkdocs.yml`:

```yaml
theme:
  name: material
  palette:
    primary: teal
    accent: amber
  features:
    - navigation.tabs
    - navigation.sections
    - search.highlight
```

See the [Material for MkDocs documentation](https://squidfunk.github.io/mkdocs-material/) for more customization options.

### Adding Pages

To add a new page to the documentation:

1. Create a new Markdown file in the `docs/` directory
2. Add it to the navigation in `mkdocs.yml`:

```yaml
nav:
  - Home: index.md
  - Your New Page: new_page.md
```

### Adding Images

Place images in a `docs/assets/` directory and reference them in Markdown:

```markdown
![Alt text](assets/image.png)
```

## Troubleshooting

### Workflow Fails

If the GitHub Actions workflow fails:

1. Check the workflow logs in the **Actions** tab
2. Common issues:
   - Missing dependencies in `docs.yml`
   - Incorrect file paths in `mkdocs.yml`
   - Syntax errors in Markdown files

### Documentation Not Updating

If the documentation doesn't update after pushing:

1. Check that the workflow completed successfully
2. Clear your browser cache
3. Wait a few minutes for GitHub Pages to update
4. Verify the `gh-pages` branch was created and updated

### 404 Error

If you get a 404 error when visiting the documentation:

1. Verify GitHub Pages is enabled in repository settings
2. Check that the `gh-pages` branch exists
3. Ensure the `site_url` in `mkdocs.yml` matches your GitHub Pages URL

## Updating the Documentation

To update the documentation:

1. Edit the Markdown files in the `docs/` directory
2. Preview changes locally with `mkdocs serve`
3. Commit and push your changes:

```bash
git add docs/
git commit -m "Update documentation"
git push origin main
```

The GitHub Actions workflow will automatically rebuild and deploy the updated documentation.

## Advanced Configuration

### Custom Domain

To use a custom domain for your documentation:

1. Add a `CNAME` file to the `docs/` directory with your domain name
2. Configure your DNS provider to point to GitHub Pages
3. Update `site_url` in `mkdocs.yml` to your custom domain

### Versioning

To maintain multiple versions of the documentation, consider using [mike](https://github.com/jimporter/mike):

```bash
pip install mike
mike deploy --push --update-aliases 1.0 latest
```

### Search Configuration

The documentation includes built-in search. To customize it:

```yaml
plugins:
  - search:
      lang: en
      separator: '[\s\-\.]+'
```

## Resources

- [MkDocs Documentation](https://www.mkdocs.org/)
- [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)
- [GitHub Pages Documentation](https://docs.github.com/en/pages)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)

## Support

If you encounter issues with the documentation setup:

1. Check the [MkDocs documentation](https://www.mkdocs.org/)
2. Review the [Material for MkDocs documentation](https://squidfunk.github.io/mkdocs-material/)
3. Open an issue on the SOLWEIG-GPU GitHub repository
