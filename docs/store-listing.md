# Chrome Web Store listing copy

Paste-ready text for the dashboard fields.

## Summary (132 characters max)

    Replaces everyday words with harder ones as you read. Hover to see the original, click to revert. Runs entirely on-device.

## Detailed description

    Complexity Injector builds your vocabulary from the pages you already read.

    As you browse, it quietly replaces everyday words with harder equivalents —
    "talkative" becomes "garrulous", "avoid" becomes "eschew", "a lack of"
    becomes "a dearth of". Hover over any changed word to see the definition and
    the word it replaced. Click it to put the original back.

    There is nothing to study and no list to work through. You meet the words in
    real sentences, in context, while reading what you were going to read anyway.

    HOW IT DECIDES

    A substitution is only shown if a language model judges that it genuinely
    fits the sentence. The model runs on your own machine, using your GPU where
    available. It is deliberately cautious: it declines far more substitutions
    than it accepts, because a wrong word on the page costs more than a missed
    opportunity.

    On held-out text it accepts 41% of candidate substitutions at 90% precision.

    PRIVACY

    Nothing leaves your browser. The model is bundled with the extension, so
    there is no server, no account, no API key, and no network request while it
    runs. No analytics, no telemetry.

    VOCABULARY

    The target words are GregMat's 900-word GRE list. If you are preparing for
    the GRE, this is a way to keep meeting those words outside of study time.

    NOTE ON SIZE

    This extension is large because the language model ships inside it rather
    than running on someone else's server. That is the tradeoff that keeps your
    browsing private.

## Category

    Education

## Single purpose

    Replaces words on web pages with more advanced synonyms to teach vocabulary
    in context.

## Privacy form

**Remote code:** No, I am not using Remote code

**Data collected:** none — tick no boxes. Certify all three disclosures.

**Privacy policy URL:** https://pu-suo.github.io/complexity-injector/privacy.html

### Single purpose description

```
Complexity Injector replaces everyday words on web pages with more advanced synonyms, so that the reader learns vocabulary in context while reading what they were already going to read. Hovering a replaced word shows its definition and the word it replaced; clicking it restores the original.

That is the extension's only function. It does not summarise, translate, annotate, collect analytics, or modify pages in any other way. All processing happens on the user's own device using a language model bundled inside the extension package.
```

### offscreen justification

```
The extension decides whether each substitution actually fits the sentence using a neural network bundled with the package. The model needs WebGPU where it is available, and a persistent inference session that occupies several hundred megabytes of memory.

A background service worker cannot reliably access WebGPU and is terminated when idle, which would force the model to be reloaded repeatedly and make the extension unusable. An offscreen document is the supported way to host a long-lived inference session.

The offscreen document displays no interface and communicates only with the extension's own service worker. It makes no network requests.
```

### storage justification

```
Stores a single boolean value: whether the user has switched the extension on or off.

chrome.storage.sync is used so that this preference follows the user's Chrome profile between their own devices, which is the behaviour users expect from an on/off switch.

No page content, browsing history, URLs, or personal information is stored. Nothing else is written to storage.
```

### Host permission justification

```
The extension substitutes words on whatever page the user is reading. It cannot know in advance which sites those will be, because the user chooses them, so it requires access to pages generally rather than to a fixed list.

The access is used only to read visible text and replace words within it. Page text is processed entirely on the user's own device by a model bundled in the package. It is never transmitted, stored, or logged, and the extension makes no network requests at runtime.

The content script deliberately does not touch form fields, text areas, editable regions, code blocks, or link text, and every substitution can be reversed with a single click. A global switch disables it entirely and restores the page.
```

### Remote code justification (if asked)

```
No remote code is used. All JavaScript, WebAssembly, and model weights are included in the extension package.

The ONNX Runtime Web library and its .wasm binaries are bundled under lib/ort/ and loaded through chrome.runtime.getURL. Every fetch() call in the extension resolves to a packaged file. No script is retrieved from a remote server, and the extension's content security policy (script-src 'self' 'wasm-unsafe-eval') would block it if anything attempted to.
```

