# Thesis Showcase — Deploy to production

Single-page static app (HTML + CSS + JS). No build step. Deploy with Netlify.

## Deploy with Netlify

1. Push this repo to GitHub.
2. Go to [netlify.com](https://netlify.com) → **Add new site** → **Import an existing project**.
3. Connect GitHub and select this repo.
4. Netlify reads the root `netlify.toml`: **Publish directory** = `showcase`. Leave build command empty (or use `echo 'ok'`).
5. Deploy. Your site will be at `https://<random>.netlify.app` (you can set a custom domain in Site settings → Domain management).

## Other static hosts (optional)

- **GitHub Pages:** Use a `gh-pages` branch with the contents of `showcase/` at root; in repo Settings → Pages, set source to branch `gh-pages`, folder `/`.
- **S3, Cloudflare Pages, etc.:** Upload the contents of the `showcase` folder so `index.html` is at the site root.

## Dependencies

- **highlight.js** is loaded from cdnjs (CSS + JS). No install; works in production as long as the CDN is reachable.
- No API or backend for the static page itself. The **Live Analysis** tab only works when the thesis API is run **locally** (e.g. `USE_REAL_ML=1 uvicorn api.main:app --reload` from the repo root) or on a GPU-backed server; see the repo root for running the API.

## Custom domain

In Netlify: Site settings → Domain management → Add custom domain, then follow the DNS instructions.
