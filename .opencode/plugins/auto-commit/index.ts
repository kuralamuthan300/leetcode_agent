import { Plugin } from "@opencode/plugin"
import { execFileSync } from "node:child_process"

const READONLY_TOOLS = new Set([
  "read",
  "glob",
  "grep",
  "webfetch",
  "websearch",
  "todoread",
  "todowrite",
])

function runGit(root: string, args: string[]): string {
  return execFileSync("git", args, {
    cwd: root,
    encoding: "utf8",
    timeout: 30_000,
    env: { ...process.env, GIT_TERMINAL_PROMPT: "0", GIT_OPTIONAL_LOCKS: "0" },
  }).toString()
}

export default Plugin.define({
  id: "auto-commit",
  async setup(ctx) {
    const root = ctx.location.directory
    const dirty = new Set<string>()
    let committing = false
    const timers = new Map<string, ReturnType<typeof setTimeout>>()

    function scheduleCommit(sessionID: string, delayMs: number, reason: string) {
      const existing = timers.get(sessionID)
      if (existing) clearTimeout(existing)
      const timer = setTimeout(() => {
        timers.delete(sessionID)
        void tryCommit(sessionID, reason).catch((e) => console.error(`[auto-commit] ${e}`))
      }, delayMs)
      timers.set(sessionID, timer)
    }

    async function tryCommit(sessionID: string, reason: string) {
      if (committing) {
        scheduleCommit(sessionID, 5000, reason)
        return
      }
      if (!dirty.has(sessionID)) return
      dirty.delete(sessionID)
      committing = true
      try {
        let status = ""
        try {
          status = runGit(root, ["status", "--porcelain"]).trim()
        } catch (e) {
          console.error(`[auto-commit] git status failed: ${e}`)
          return
        }
        if (!status) return
        try {
          runGit(root, ["add", "-A"])
        } catch (e) {
          console.error(`[auto-commit] git add failed: ${e}`)
          return
        }
        const short = sessionID.replace(/^ses_?/, "").slice(0, 8)
        const msg = `opencode: auto-commit ${short} ${new Date().toISOString()} (${reason})`
        try {
          runGit(root, ["commit", "-m", msg])
          console.log(`[auto-commit] committed (${reason}): ${msg}`)
        } catch (e) {
          console.error(`[auto-commit] git commit failed (nothing to commit?): ${e}`)
        }
      } finally {
        committing = false
      }
    }

    await ctx.tool.hook("execute.after", async (event) => {
      if (event.status !== "completed") return
      if (READONLY_TOOLS.has(event.tool)) return
      if (event.tool === "shell") {
        const cmd = JSON.stringify((event.input as Record<string, unknown> | undefined) ?? {})
        if (cmd.includes("git commit") || cmd.includes("auto-commit")) return
      }
      dirty.add(event.sessionID)
      scheduleCommit(event.sessionID, 15_000, `debounce:${event.tool}`)
    })

    const controller = new AbortController()
    void (async () => {
      try {
        for await (const event of ctx.event.subscribe({ signal: controller.signal })) {
          const type = (event as { type?: string }).type
          if (type !== "session.idle" && type !== "session.execution.succeeded" && type !== "session.execution.failed") {
            continue
          }
          const sessionID = (event as { data?: { sessionID?: string } }).data?.sessionID
          if (!sessionID || !dirty.has(sessionID)) continue
          const loc = (event as { location?: { directory?: string } }).location?.directory
          if (loc && loc !== root) continue
          await tryCommit(sessionID, type)
        }
      } catch (e) {
        if ((e as Error)?.name !== "AbortError") console.error(`[auto-commit] event loop ended: ${e}`)
      }
    })()

    return () => {
      controller.abort()
      for (const t of timers.values()) clearTimeout(t)
      timers.clear()
    }
  },
})
