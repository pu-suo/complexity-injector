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

## Permission justifications

**host permission (all sites)**

    The extension substitutes words on whatever page the user is reading. It
    cannot know in advance which sites those will be, so it needs access to the
    pages the user chooses to visit. The access is used only to read visible
    text and replace words within it. No page content is transmitted anywhere.

**offscreen**

    The substitution decision is made by a neural network bundled with the
    extension. It requires WebGPU and a persistent inference session of several
    hundred megabytes. A background service worker cannot reliably access WebGPU
    and is terminated when idle, which would reload the model repeatedly. An
    offscreen document is the supported way to host it.

**storage**

    Stores a single boolean: whether the extension is switched on. No user data
    is stored.

## Data usage disclosures

Tick: does NOT collect or use personally identifiable information, health
information, financial information, authentication information, personal
communications, location, web history, or user activity.

"Website content" — the extension reads page text, but processes it locally and
never transmits it. Declare that it is not collected.

Certify all three compliance checkboxes.
