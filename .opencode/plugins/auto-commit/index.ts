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

const WIP_PREFIX = "opencode/"
const MAX_WIP_BRANCHES = 20

function runGit(root: string, args: string[], extraEnv?: Record<string, string>): string {
  return execFileSync("git", args, {
    cwd: root,
    encoding: "utf8",
    timeout: 30_000,
    env: { ...process.env, GIT_TERMINAL_PROMPT: "0", GIT_OPTIONAL_LOCKS: "0", ...extraEnv },
  }).toString()
}

function gitOk(root: string, args: string[]): string | undefined {
  try {
    return runGit(root, args).trim()
  } catch {
    return undefined
  }
}

function shortID(sessionID: string): string {
  return sessionID.replace(/^ses_?/, "").slice(0, 8)
}

function todayStamp(): string {
  const d = new Date()
  const pad = (n: number) => String(n).padStart(2, "0")
  return `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}`
}

function findWipBranch(root: string, short: string): string | undefined {
  const out = gitOk(root, ["for-each-ref", "--format=%(refname:short)", `refs/heads/${WIP_PREFIX}*-${short}`])
  return out?.split("\n").map((s) => s.trim()).filter(Boolean)[0]
}

export default Plugin.define({
  id: "auto-commit",
  async setup(ctx) {
    const root = ctx.location.directory
    const dirty = new Set<string>()
    let committing = false
    const timers = new Map<string, ReturnType<typeof setTimeout>>()

    function pruneMergedWip(except?: string) {
      const merged = gitOk(root, ["branch", "--merged", "HEAD", "--list", `${WIP_PREFIX}*`, "--format=%(refname:short)"])
      for (const b of (merged ?? "").split("\n").map((s) => s.trim()).filter(Boolean)) {
        if (b === except) continue
        try {
          runGit(root, ["branch", "-d", b])
          console.log(`[auto-commit] pruned merged WIP branch ${b}`)
        } catch (e) {
          console.error(`[auto-commit] could not prune ${b}: ${e}`)
        }
      }
      const all = (gitOk(root, ["for-each-ref", "--format=%(refname:short)", `refs/heads/${WIP_PREFIX}*`]) ?? "")
        .split("\n")
        .map((s) => s.trim())
        .filter(Boolean)
      if (all.length > MAX_WIP_BRANCHES) {
        console.log(`[auto-commit] ${all.length} WIP branches retained (over cap of ${MAX_WIP_BRANCHES}); oldest unmerged branches left for manual cleanup`)
      }
    }

    async function tryWipCommit(sessionID: string, reason: string) {
      if (committing) {
        scheduleCommit(sessionID, 5000, reason)
        return
      }
      if (!dirty.has(sessionID)) return
      dirty.delete(sessionID)
      committing = true
      try {
        const status = gitOk(root, ["status", "--porcelain"]) ?? ""
        if (!status.trim()) return
        const short = shortID(sessionID)
        const branch = findWipBranch(root, short) ?? `${WIP_PREFIX}${todayStamp()}-${short}`
        try {
          runGit(root, ["add", "-A"])
        } catch (e) {
          console.error(`[auto-commit] git add failed: ${e}`)
          return
        }
        const tree = runGit(root, ["write-tree"]).trim()
        const commitArgs = ["commit-tree", tree]
        const existingTip = gitOk(root, ["rev-parse", "--verify", branch])
        if (existingTip) {
          commitArgs.push("-p", existingTip)
        } else {
          const head = gitOk(root, ["rev-parse", "--verify", "HEAD"])
          if (head) commitArgs.push("-p", head)
        }
        const msg = `opencode wip ${short} ${new Date().toISOString()} (${reason})`
        let commit: string
        try {
          commit = runGit(root, [...commitArgs, "-m", msg]).trim()
        } catch (e) {
          // Fallback when the repo has no git identity configured.
          if (gitOk(root, ["config", "user.name"]) && gitOk(root, ["config", "user.email"])) throw e
          console.log("[auto-commit] no git identity configured, committing as opencode fallback")
          commit = execFileSync("git", [...commitArgs, "-m", msg], {
            cwd: root,
            encoding: "utf8",
            timeout: 30_000,
            env: {
              ...process.env,
              GIT_TERMINAL_PROMPT: "0",
              GIT_AUTHOR_NAME: "opencode",
              GIT_AUTHOR_EMAIL: "opencode@local",
              GIT_COMMITTER_NAME: "opencode",
              GIT_COMMITTER_EMAIL: "opencode@local",
            },
          })
            .toString()
            .trim()
        }
        runGit(root, ["update-ref", `refs/heads/${branch}`, commit])
        try {
          runGit(root, ["reset"])
        } catch {
          // Unborn HEAD or similar; index state is harmless, files are untouched.
        }
        const stat = gitOk(root, ["show", "--stat", "--oneline", commit]) ?? commit
        console.log(`[auto-commit] WIP commit on ${branch} (${reason}):\n${stat}`)
        pruneMergedWip(branch)
      } catch (e) {
        console.error(`[auto-commit] WIP commit failed: ${e}`)
      } finally {
        committing = false
      }
    }

    function scheduleCommit(sessionID: string, delayMs: number, reason: string) {
      const existing = timers.get(sessionID)
      if (existing) clearTimeout(existing)
      const timer = setTimeout(() => {
        timers.delete(sessionID)
        void tryWipCommit(sessionID, reason).catch((e) => console.error(`[auto-commit] ${e}`))
      }, delayMs)
      timers.set(sessionID, timer)
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

    await ctx.command.transform((editor) => {
      editor.add({
        name: "review",
        description: "Show this session's auto-commit WIP branch diff for review",
        execute: async ({ sessionID, prompt, delivery }) => {
          const short = shortID(sessionID)
          const branch = findWipBranch(root, short)
          if (!branch) {
            await ctx.session.prompt({
              ...prompt,
              sessionID,
              delivery,
              text: "There is no auto-commit WIP branch for this session yet (no changes have been committed). Tell the user there is nothing to review.",
            })
            return
          }
          const base = gitOk(root, ["merge-base", "HEAD", branch]) ?? gitOk(root, ["rev-list", "--max-parents=0", branch])
          const stat = base ? (gitOk(root, ["diff", base, branch, "--stat"]) ?? "(empty diff)") : "(could not determine base)"
          const commits = base ? (gitOk(root, ["log", "--oneline", `${base}..${branch}`]) ?? "") : ""
          await ctx.session.prompt({
            ...prompt,
            sessionID,
            delivery,
            text: `Review the session's auto-commit WIP branch ${branch} (${commits.split("\n").filter(Boolean).length} commit(s)):\n\nCommits:\n${commits || "(none)"}\n\nDiff stat vs base ${base?.slice(0, 8)}:\n${stat}\n\nSummarize the changes per file for the user so they can decide whether to merge them. Do not commit anything.`,
          })
        },
      })
    })

    pruneMergedWip()

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
          await tryWipCommit(sessionID, type)
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
