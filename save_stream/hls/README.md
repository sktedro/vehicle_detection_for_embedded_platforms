# Brief

- Simply saves the stream without encoding. Requires almost no CPU. Uses this
  command: `ffmpeg -i <URL> -loglevel error -c copy -an output.ts`
- Config is in a simple CSV format: <name>:<url.m3u8>

# How to stop?!

- For the save stream script: `touch stop` or use any way to create a `stop` file in the folder
- For the processing script: The same, but `stop_process` (or simply `stop_p*`)
