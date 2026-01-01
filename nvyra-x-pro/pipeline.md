

claim - rust - nemtron orchestrator it can decide a couple things
 - claim extraction (via Nemotron 30B- A3B- FP8 )
 - web search (via Tavily with key rotation)
 -internal cache search
    - starts of for sentence similarity/keyword matching in qdrant on my contabo vm
    - then when a match is found it like goes to turso finds the relevant metadata and then gets the text from backblaze (with all the details and stuff)

2. Then the Nvidia Nemotron 30B-A3B-FP8 model like gets the Thentext and the query and reasons if the text and stuff supports the claim and it has enough information or if it needs to do further cache search/web searching. 
3. While all of this happen - 2 more things are going on in parralel. Firstly, my fintuned model for disinformation detection is running on the text (500Ms on H200) and then embeddings are being calculated for the text by like a small but powerful embeddings model and then features will be compted from their embeddings.

4. The Nemotron Ochestrator sees the text and decides whether a specalist reasoning model needs to be employed to check nemotrons answer -- available at this Hugging Face repo Feargal/nvyra-x-reasoning My custom 557M paramater reasoning model will then combine the three (the nemotron output, the extracted features and the qwen disinformation detection model) outputs together and reason about it and output it to the user. It will also contain a confidence threshold and also prompt guarding to make sure it doesnt output harmful stuff, along with citations of the URL's it looked at and the sources used.

5. If a web search was called then it will save the text and metadata and embedd it and everything and check for duplicates in turso and if not then it will add it to qdrant and backblaze. Maybe this will be a seperate modal app and the user will not see this happening. It will have its own H200 for doing all of this stuff and the goal is to build up a cache so it doesnt have to pay for tavily web searching as much. Looking for like a 94% cache hit rate hopefully. 
