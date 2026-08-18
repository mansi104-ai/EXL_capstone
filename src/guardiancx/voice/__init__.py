"""The live-call layer.

Everything needed to run the vulnerability pipeline *during* a phone call rather
than after it:

* `prosody` — what the audio says that the words do not: distress in the voice.
* `endpointing` — deciding when the customer has actually finished speaking.
* `live_pii` — redacting personal data out of a transcript as it arrives.

None of it requires a native audio dependency: the analysis is standard-library
signal processing over the WAV bytes the browser already captures.
"""
