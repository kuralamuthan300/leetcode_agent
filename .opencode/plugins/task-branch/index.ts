import { Plugin } from "@opencode/plugin"
import { execFileSync } from "node:child_process"

const BRANCH_PREFIX = "task/"
const SLUG_RE = /^[a-z0-9]+(?:-[a-z0-9]+)*$/
const MAX_SLUG_LEN = 60

function runGit(root: string, args: string[]): string {
  return execFileSync("git", args, {
    cwd: root,
    encoding: "utf8",
    timeout: 30_000,
    env: { ...process.env, GIT_TERMINAL_PROMPT: "0" },
  }).toString()
}

function gitOk(root: string, args: string[]): string | undefined {
  try {
    return runGit(root, args).trim()
  } catch {
    return undefined
  }
}

function validateSlug(slug: string): void {
  if (!slug || slug.length > MAX_SLUG_LEN || !SLUG_RE.test(slug)) {
    throw new Error(
      `invalid slug ${JSON.stringify(slug)}: use lowercase a-z0-9 with dashes only, max ${MAX_SLUG_LEN} chars (e.g. step-4-sandbox)`
    )
  }
}

export type ChangedFile = { status: string; path: string }

export function parseNameStatus(output: string): ChangedFile[] {
  return output
    .split("\n")
    .map((l) => l.trim())
    .filter(Boolean)
    .filter((l) => l !== "(no files)")
    .map((line) => {
      const parts = line.split("\t")
      const code = parts[0] ?? ""
      const status = code.replace(/[0-9]/g, "") || code
      const filePath = (parts[parts.length - 1] ?? "").replace(/^"|"$/g, "")
      return { status, path: filePath }
    })
}

export function buildVisualSection(files: ChangedFile[], base: string, branch: string): string {
  const lines = ["", "Visual review:"]
  if (!files.length) {
    lines.push("- no changed files.")
    return lines.join("\n")
  }
  for (const f of files) {
    lines.push(`- [${f.status}] ${f.path}`)
  }
  lines.push(`- side-by-side: run \`git difftool --dir-diff ${base}...${branch}\``)
  return lines.join("\n")
}

export type TaskLiveInfo = {
  slug: string
  base: string
  branch: string
  current: boolean
  dirty: boolean
  ahead: number
}

export function formatTaskRow(t: TaskLiveInfo): string {
  const flags = [t.current ? "current" : "not-current", t.dirty ? "dirty:yes" : "clean", `ahead:${t.ahead}`].join(" ")
  return `- ${t.slug} [${flags}]\n  base: ${t.base} branch: ${t.branch}`
}

export function parseBaseDescription(desc: string | undefined): string | undefined {
  if (!desc) return undefined
  const m = desc.match(/^base:\s*(.+?)\s*$/m)
  return m?.[1]?.trim() || undefined
}

export default Plugin.define({
  id: "task-branch",
  async setup(ctx) {
    const root = ctx.location.directory
    const opts = (ctx.options ?? {}) as { prefix?: string }
    const prefix = typeof opts.prefix === "string" && opts.prefix ? opts.prefix : BRANCH_PREFIX

    const branchName = (slug: string) => `${prefix}${slug}`
    const slugOf = (branch: string) =>
      branch.startsWith(prefix) ? branch.slice(prefix.length) : undefined

    function currentBranch(): string | undefined {
      const b = gitOk(root, ["branch", "--show-current"])
      return b || undefined
    }

    function getBase(branch: string): string | undefined {
      const desc = gitOk(root, ["config", `branch.${branch}.description`])
      return parseBaseDescription(desc)
    }

    function setBase(branch: string, base: string): void {
      runGit(root, ["config", `branch.${branch}.description`, `base: ${base}`])
    }

    function branchExists(branch: string): boolean {
      return !!gitOk(root, ["rev-parse", "--verify", branch])
    }

    function isClean(): string {
      return (gitOk(root, ["status", "--porcelain"]) ?? "").trim()
    }

    async function doStart(slug: string): Promise<string> {
      validateSlug(slug)
      const branch = branchName(slug)
      const dirty = isClean()
      if (dirty) {
        throw new Error(`refusing to start: working tree has uncommitted changes. Commit or stash first, then /task-start ${slug}.`)
      }
      const base = currentBranch()
      if (!base) {
        throw new Error("could not determine current branch (detached HEAD?). Checkout a branch first, then /task-start <slug>.")
      }
      if (branchExists(branch)) {
        throw new Error(`branch ${branch} already exists. Pick another slug or merge/delete it first.`)
      }
      try {
        runGit(root, ["checkout", "-b", branch])
      } catch (e) {
        throw new Error(`git checkout -b ${branch} failed: ${e}`)
      }
      setBase(branch, base)
      return `started task ${slug}\nbase: ${base}\nbranch: ${branch}\nYou are now on ${branch} (branched from ${base}). Give the requirement and all edits will land here.`
    }

    async function doDiff(slug: string): Promise<string> {
      validateSlug(slug)
      const branch = branchName(slug)
      if (!branchExists(branch)) {
        throw new Error(`branch ${branch} does not exist.`)
      }
      const base = getBase(branch)
      if (!base) {
        throw new Error(`no base recorded for ${branch} (missing branch description "base: <name>").`)
      }
      const mergeBase = gitOk(root, ["merge-base", base, branch]) ?? base
      const stat = gitOk(root, ["diff", `${base}...${branch}`, "--stat"]) || "(empty diff — no committed changes yet)"
      const namesRaw = gitOk(root, ["diff", `${base}...${branch}`, "--name-status"]) || "(no files)"
      const commits = gitOk(root, ["log", "--oneline", `${base}..${branch}`]) || "(no commits yet)"
      const cur = currentBranch()
      let uncommitted = ""
      if (cur === branch) {
        const u = isClean()
        if (u) uncommitted = `\n\nWARNING: uncommitted changes on ${branch} (not in branch diff):\n${u}\nCommit them first for an accurate review.`
      }
      const visual = buildVisualSection(parseNameStatus(namesRaw), base, branch)
      return `diff ${base}...${branch} (merge-base ${mergeBase.slice(0, 8)})\n\nCommits (${base}..${branch}):\n${commits}\n\nStat:\n${stat}\n\nFiles:\n${namesRaw}${uncommitted}${visual}`
    }

    async function doMerge(slug: string, squash: boolean): Promise<string> {
      validateSlug(slug)
      const branch = branchName(slug)
      if (!branchExists(branch)) {
        throw new Error(`branch ${branch} does not exist; nothing to merge.`)
      }
      const base = getBase(branch)
      if (!base) {
        throw new Error(`no base recorded for ${branch} (missing branch description "base: <name>").`)
      }
      const cur = currentBranch()
      if (cur === branch && isClean()) {
        // clean task branch — safe to leave it
      } else if (cur === branch) {
        const u = isClean()
        throw new Error(`task branch ${branch} has uncommitted changes:\n${u}\nCommit them first, then /task-merge ${slug}.`)
      }
      try {
        runGit(root, ["checkout", base])
      } catch (e) {
        throw new Error(`checkout ${base} failed: ${e}. Commit or stash changes first.`)
      }
      const baseDirty = isClean()
      if (baseDirty) {
        throw new Error(`${base} has uncommitted changes:\n${baseDirty}\nCommit or stash first, then /task-merge ${slug}.`)
      }
      try {
        if (squash) runGit(root, ["merge", "--squash", branch])
        else runGit(root, ["merge", "--no-ff", branch])
      } catch (e) {
        throw new Error(`merge ${branch} into ${base} failed (resolve conflicts, then commit): ${e}`)
      }
      if (squash) {
        return `squashed ${branch} into ${base} (staged, NOT committed). Review with git status / git diff --cached, then commit. Finish cleanup with /task-merge ${slug} --cleanup (or: git branch -D ${branch}) after committing.`
      }
      try {
        runGit(root, ["branch", "-d", branch])
      } catch {
        // leave for manual cleanup if not fully merged
      }
      return `merged ${branch} into ${base} (--no-ff). Branch deleted. Verify with git log --oneline -3.`
    }

    async function doCleanup(slug: string): Promise<string> {
      validateSlug(slug)
      const branch = branchName(slug)
      const base = getBase(branch) ?? "(unknown base)"
      const cur = currentBranch()
      if (cur === branch) {
        // move off the branch before deleting it
        const fallback = typeof base === "string" && base.startsWith("(unknown") ? undefined : (base as string)
        if (fallback && branchExists(fallback)) {
          runGit(root, ["checkout", fallback])
        } else {
          throw new Error(`cannot delete ${branch} while it is checked out. Checkout its base first, then retry.`)
        }
      }
      try {
        runGit(root, ["branch", "-D", branch])
      } catch (e) {
        return `cleanup failed for ${slug}: ${e}`
      }
      return `cleaned up task ${slug}: branch ${branch} deleted (was based on ${base}). You are now on ${currentBranch()}.`
    }

    async function doList(): Promise<string> {
      const out = gitOk(root, ["for-each-ref", "--format=%(refname:short)", `refs/heads/${prefix}*`]) ?? ""
      const branches = out.split("\n").map((s) => s.trim()).filter(Boolean)
      if (!branches.length) return "no active tasks (no task/* branches). Start one with /task-start <slug>."
      const cur = currentBranch()
      const rows: TaskLiveInfo[] = branches.map((branch) => {
        const slug = slugOf(branch) ?? branch
        const base = getBase(branch) ?? "(unknown — no branch description)"
        let ahead = 0
        if (base && !base.startsWith("(unknown") && branchExists(base)) {
          const count = gitOk(root, ["rev-list", "--count", `${base}..${branch}`])
          ahead = Number.parseInt(count ?? "0", 10) || 0
        }
        const dirty = cur === branch ? !!isClean() : false
        return { slug, base, branch, current: cur === branch, dirty, ahead }
      })
      return `active tasks (${rows.length}):\n` + rows.map(formatTaskRow).join("\n")
    }

    await ctx.tool.transform((editor) => {
      editor.namespace({ name: "task", description: "Branch-per-task workflow (no worktrees)" })
      editor.add({
        name: "start",
        description: "Start a task: create task/<slug> from current branch, record base in branch description",
        input: {
          type: "object",
          properties: { slug: { type: "string" } },
          required: ["slug"],
          additionalProperties: false,
        },
        options: { namespace: "task", codemode: true },
        execute: async (input) => {
          const { slug } = input as { slug: string }
          try {
            const out = await doStart(slug)
            return { content: out }
          } catch (e) {
            return { content: `task_start failed: ${(e as Error).message}` }
          }
        },
      })
      editor.add({
        name: "diff",
        description: "Show diff of task branch vs its recorded base branch",
        input: {
          type: "object",
          properties: { slug: { type: "string" } },
          required: ["slug"],
          additionalProperties: false,
        },
        options: { namespace: "task", codemode: true },
        execute: async (input) => {
          const { slug } = input as { slug: string }
          try {
            return { content: await doDiff(slug) }
          } catch (e) {
            return { content: `task_diff failed: ${(e as Error).message}` }
          }
        },
      })
      editor.add({
        name: "merge",
        description: "Merge task branch back into its base branch (squash by default)",
        input: {
          type: "object",
          properties: { slug: { type: "string" }, squash: { type: "boolean" } },
          required: ["slug"],
          additionalProperties: false,
        },
        options: { namespace: "task", codemode: true },
        execute: async (input) => {
          const { slug, squash } = input as { slug: string; squash?: boolean }
          try {
            return { content: await doMerge(slug, squash !== false) }
          } catch (e) {
            return { content: `task_merge failed: ${(e as Error).message}` }
          }
        },
      })
      editor.add({
        name: "list",
        description: "List all active task branches with base/dirty status",
        input: {
          type: "object",
          properties: {},
          additionalProperties: false,
        },
        options: { namespace: "task", codemode: true },
        execute: async () => {
          try {
            return { content: await doList() }
          } catch (e) {
            return { content: `task_list failed: ${(e as Error).message}` }
          }
        },
      })
    })

    await ctx.command.transform((editor) => {
      editor.add({
        name: "task-start",
        description: "Start task <slug>: branch task/<slug> from current branch",
        execute: async ({ sessionID, prompt, delivery }) => {
          const slug = prompt.text.trim().split(/\s+/)[0] ?? ""
          if (!slug) {
            await ctx.session.prompt({
              ...prompt, sessionID, delivery,
              text: "Usage: /task-start <slug> (e.g. /task-start step-4-sandbox). Slug: lowercase a-z0-9 with dashes.",
            })
            return
          }
          try {
            const out = await doStart(slug)
            await ctx.session.prompt({ ...prompt, sessionID, delivery, text: `${out}\n\nProceed with the task on this branch.` })
          } catch (e) {
            await ctx.session.prompt({ ...prompt, sessionID, delivery, text: `task-start failed: ${(e as Error).message}` })
          }
        },
      })
      editor.add({
        name: "task-diff",
        description: "Show diff of task <slug> vs its base branch",
        execute: async ({ sessionID, prompt, delivery }) => {
          const slug = prompt.text.trim().split(/\s+/)[0] ?? ""
          if (!slug) {
            await ctx.session.prompt({ ...prompt, sessionID, delivery, text: "Usage: /task-diff <slug>" })
            return
          }
          try {
            const out = await doDiff(slug)
            await ctx.session.prompt({ ...prompt, sessionID, delivery, text: `${out}\n\nSummarize per file so the user can decide whether to merge. Do not commit anything.` })
          } catch (e) {
            await ctx.session.prompt({ ...prompt, sessionID, delivery, text: `task-diff failed: ${(e as Error).message}` })
          }
        },
      })
      editor.add({
        name: "task-merge",
        description: "Merge task <slug> into its base branch (squash default; --no-squash for merge commit; --cleanup to delete branch after squash)",
        execute: async ({ sessionID, prompt, delivery }) => {
          const parts = prompt.text.trim().split(/\s+/).filter(Boolean)
          const slug = parts[0] ?? ""
          if (!slug) {
            await ctx.session.prompt({ ...prompt, sessionID, delivery, text: "Usage: /task-merge <slug> [--no-squash] [--cleanup]" })
            return
          }
          try {
            const out = parts.includes("--cleanup")
              ? await doCleanup(slug)
              : await doMerge(slug, !parts.includes("--no-squash"))
            await ctx.session.prompt({ ...prompt, sessionID, delivery, text: out })
          } catch (e) {
            await ctx.session.prompt({ ...prompt, sessionID, delivery, text: `task-merge failed: ${(e as Error).message}` })
          }
        },
      })
      editor.add({
        name: "task-list",
        description: "List all active task branches with base/dirty status",
        execute: async ({ sessionID, prompt, delivery }) => {
          try {
            const out = await doList()
            await ctx.session.prompt({ ...prompt, sessionID, delivery, text: out })
          } catch (e) {
            await ctx.session.prompt({ ...prompt, sessionID, delivery, text: `task-list failed: ${(e as Error).message}` })
          }
        },
      })
      editor.add({
        name: "task-abort",
        description: "Abort task <slug>: checkout base and delete task branch without merging",
        execute: async ({ sessionID, prompt, delivery }) => {
          const slug = prompt.text.trim().split(/\s+/)[0] ?? ""
          if (!slug) {
            await ctx.session.prompt({ ...prompt, sessionID, delivery, text: "Usage: /task-abort <slug>" })
            return
          }
          try {
            const out = await doCleanup(slug)
            await ctx.session.prompt({ ...prompt, sessionID, delivery, text: out })
          } catch (e) {
            await ctx.session.prompt({ ...prompt, sessionID, delivery, text: `task-abort failed: ${(e as Error).message}` })
          }
        },
      })
    })
  },
})
