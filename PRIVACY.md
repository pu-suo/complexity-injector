# Privacy Policy

**Complexity Injector** — last updated 18 August 2026

## Summary

This extension collects nothing, transmits nothing, and contacts no server.

## What it does with page content

The extension reads visible text on pages you visit in order to find words it
can replace. That text is processed entirely inside your browser by a language
model bundled with the extension. It is never sent anywhere, never written to
disk, and is discarded as soon as the page is processed.

There is no analytics, no telemetry, no error reporting, and no network request
of any kind at runtime. The extension has no server component.

## What it stores

One setting: whether the extension is switched on or off. It is kept in Chrome's
`storage.sync`, which means Chrome syncs it to your own Google account so the
setting follows you between devices. Nothing else is stored, and the developer
cannot see it.

## Permissions

- **Host access to all sites** — substitution happens on whatever page you are
  reading, and the extension cannot know in advance which sites those are. This
  access is used only to read and rewrite visible text in the page.
- **Offscreen** — the language model needs a long-lived document to run in.
  A background service worker cannot hold it.
- **Storage** — the on/off setting described above.

## Data sharing

None. No data is collected, so none is sold, shared, or transferred.

## Contact

Issues and questions: https://github.com/pu-suo/complexity-injector/issues
