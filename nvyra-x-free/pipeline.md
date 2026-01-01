nvyra-x free tier

This runs on a CPU and never anything else. The sole goal of this is it needs to be extreemly acccurate for extreemly cheap. Instead of the Nemtron Orchestror it will use the Qwe 3/ 1.7B model for reasoning as it is a lot faster. 

There is no orchstator here everything just happens linearly.
Firstly, claim ectraction is done by Qwen/Qwen3-1.7B-Instruct model but maybe in a .GGUF or .onnx format for speed. Then the qwen model decides should it search the cache or web search the claims. Then it like looks up on Duck Duck Go, Bing, Google ect all for free - respecting rate limites and then gets the text back to Qwen model

Either it can do the same thing as the pro tier -internal cache search
    - starts of for sentence similarity/keyword matching in qdrant on my contabo vm
    - then when a match is found it like goes to turso finds the relevant metadata and then gets the text from backblaze (with all the details and stuff)

While all of this is going on, the Qwen 2.5 -1.5 B model which has been fintuned for disinformation detection is also running - and embedding generation are happening simultaneously. Then with the embeddings - all bge-small model lots of features are computed. 

then it uses the Qwen model to reason about the claim and the text and stuff and outputs it to the user. M custom 557M paramater reasoning model will then combine the three (the qwen output, the extracted features and the qwen disinformation detection model) outputs together and reason about it and output it to the user. It will also contain a confidence threshold and also prompt guarding to make sure it doesnt output harmful stuff.