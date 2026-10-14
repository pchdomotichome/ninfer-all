# Known gaps

Defects and shortfalls found in this fork that are not yet fixed. Each entry says what it is, how
it was found, how to see it, and what the fix would touch, so it can be picked up without
rediscovering it.

Fixed items do not stay here; the commit that fixes one removes its entry.

## Router: preset options that the `HttpServer` never receives

**Status:** open. Worked around in our deployment, not fixed in the code.

**What.** In router mode an option written only in the `[*]` section of `--models-preset` is ignored
if the single `HttpServer` is what consumes it. `--request-log-jsonl` is the one we hit: the request
log stops being written entirely, with no error.

**Cause.** There is one `HttpServer` and it is built from the process options:

- `apps/serve/main.cpp:248` — `options = parse_serve_options(argc, argv)`, the command line only
- `apps/serve/main.cpp:322` — `HttpServer server(options, ...)`
- `src/serve/http_server.cpp:296` — `request_jsonl_(options_.request_log_jsonl, ...)`

The preset does not feed that `options`. `src/serve/model_catalog.cpp` turns `[*]` into
`global_arguments`, and those are only prepended to each **model's** arguments (line 204), which
reach `parse_serve_options` → `GenerationService`. `GenerationService` never constructs an
`HttpServer`.

So every `[*]` option is correct for the Engine and invisible to the server that answers requests.

**How it misled us.** `GET /models` returns `status.args`, which is the model's argument vector, not
the server's effective options. The flag is plainly visible there and does nothing. Our own
`ExecStart` carried the matching wrong belief: "the request log is written by the router from the
preset's log".

**How to see it.**

```bash
# --request-log-jsonl in the preset's [*], absent from the command line.
curl -s -XPOST localhost:9081/v1/chat/completions \
  -d '{"model":"<one of the catalog>","messages":[{"role":"user","content":"hola"}],"max_tokens":16}'
stat -c%s <the jsonl path>                  # unchanged
ls -l /proc/<pid of ninfer-serve>/fd | grep jsonl   # nothing
```

**Workaround.** Give the flag on the command line; it is then a process option and works. This is
what our `ninfer-router.service` does.

**What a fix touches.** Before the `HttpServer` is constructed in `main.cpp`, parse the preset's
`global_arguments` and merge them into `options` with the precedence `model_catalog.cpp` already
documents (global section < model section < command line). That needs `build_catalog` to hand back
`global_arguments`. Two constraints: `host`, `port` and `api_key` are the router's own and are
assigned after parsing (`main.cpp:411-413`), so the merge must not overwrite them; and the merge
should cover only what the `HttpServer` consumes, since the Engine's options already arrive by their
own path. The alternative is to construct `JsonlRequestLog` from the Engine's options in router mode
rather than the process's.

**Not yet audited:** whether any other preset `[*]` option is consumed by the `HttpServer` and so
affected the same way.