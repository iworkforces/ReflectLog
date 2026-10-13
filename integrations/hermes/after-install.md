# Configure ReflectLog

Set `REFLECTLOG_MCP_AUTH_TOKEN` in your Hermes profile environment (for example
`$HERMES_HOME/.env`) to the same value as the ReflectLog server's
`REFLECTLOG_MCP_AUTH_TOKEN`. The server URL defaults to `http://127.0.0.1:9103/mcp`; set
`REFLECTLOG_URL` to override it.

```sh
hermes memory setup     # pick reflectlog to set the token, URL and options
hermes memory status    # check the configuration and token without a network call
```

`hermes memory setup reflectlog` only selects the provider; it does not prompt
for settings. Settings live in `$HERMES_HOME/reflectlog.json`.

Turn capture is off by default. Committed built-in memory writes are mirrored
add-only through a bounded, best-effort background queue, not a durable journal,
so pending jobs can be lost on process exit. Deleting needs the explicit
`reflectlog_forget` tool.
