# sidescatter.org

The project website: one static page, no build step. Edit `index.html`,
and the `Website` workflow publishes this folder to GitHub Pages when the
change reaches `main`.

- Download buttons point at a known release, then the page asks the GitHub
  API for the newest one (pre-releases included) and updates the links,
  sizes, and version line. If that request fails, the built-in links still
  work. When you publish a release, bump the fallback version in
  `index.html` too (search for `v0.5.0`).
- Slides are 1600×1000 WebP images in `img/`, made from app screenshots on
  public demo data. The slide list and captions are at the top of the script.
- Preview locally: `python -m http.server --directory website`.
