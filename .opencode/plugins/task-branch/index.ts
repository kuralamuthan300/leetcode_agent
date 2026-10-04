import { Plugin } from "@opencode/plugin"
import { execFileSync } from "node:child_process"
import * as fs from "node:fs"
import * as path from "node:path"

const BRANCH_PREFIX = "opencode/"
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

function taskKey(slug: string): string {
  return `task:${slug}`
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

function vscodeLink(absPath: string): string {
  return `vscode://file${encodeURI(absPath)}:1`
}

export function buildVisualSection(
  files: ChangedFile[],
  worktreeDir: string | undefined,
  base: string,
  branch: string
): string {
  const lines = ["", "Visual review (Cmd/Ctrl+click a link):"]
  if (worktreeDir) {
    lines.push(`- worktree folder: ${worktreeDir}`)
    for (const f of files) {
      if (f.status === "D") {
        lines.push(`- [D] ${f.path} (deleted in ${branch})`)
        continue
      }
      const abs = path.join(worktreeDir, f.path)
      lines.push(`- [${f.status}] ${f.path}\n  open: ${abs}\n  vscode: ${vscodeLink(abs)}`)
    }
    lines.push(
      `- side-by-side: open a second window on the worktree (\`code "${worktreeDir}"\`), or run \`git difftool --dir-diff ${base}...${branch}\``
    )
  } else {
    lines.push(`- no live worktree found for ${branch}; per-file links unavailable.`)
    lines.push(`- run \`git difftool --dir-diff ${base}...${branch}\` for a visual diff.`)
  }
  return lines.join("\n")
}

function findWorktreeDir(root: string, branch: string): string | undefined {
  const out = gitOk(root, ["worktree", "list", "--porcelain"])
  if (!out) return undefined
  for (const block of out.split("\n\n")) {
    const wt = block.match(/^worktree (.+)$/m)?.[1]
    const br = block.match(/^branch (.+)$/m)?.[1]
    if (wt && br === `refs/heads/${branch}`) return wt
  }
  return undefined
}

export type TaskLiveInfo = {
  slug: string
  base: string
  branch: string
  directory: string
  createdAt: string
  branchExists: boolean
  worktreeExists: boolean
  dirty: boolean
  ahead: number
}

export function formatTaskRow(t: TaskLiveInfo): string {
  const flags = [
    t.branchExists ? "branch:yes" : "branch:MISSING",
    t.worktreeExists ? "worktree:yes" : "worktree:gone",
    t.dirty ? "dirty:yes" : "clean",
    `ahead:${t.ahead}`,
  ].join(" ")
  return `- ${t.slug} [${flags}]\n  base: ${t.base} branch: ${t.branch}\n  dir: ${t.directory}`
}

function branchFor(slug: string): string {
  return `${BRANCH_PREFIX}${slug}`
}

function worktreeDirFor(root: string, slug: string, customRoot?: string): string {
  if (customRoot) return path.resolve(root, customRoot, slug)
  return path.resolve(root, `../${path.basename(root)}-${slug}`)
}

type TaskRecord = { base: string; branch: string; directory: string; createdAt: string }

export default Plugin.define({
  id: "task-branch",
  async setup(ctx) {
    const root = ctx.location.directory
    const opts = (ctx.options ?? {}) as { root?: string; prefix?: string }
    const prefix = typeof opts.prefix === "string" && opts.prefix ? opts.prefix : BRANCH_PREFIX

    const branchName = (slug: string) => `${prefix}${slug}`

    async function doStart(sessionID: string, slug: string): Promise<string> {
      validateSlug(slug)
      const branch = branchName(slug)
      const dir = worktreeDirFor(root, slug, opts.root)

      const status = gitOk(root, ["status", "--porcelain"]) ?? ""
      if (status.trim()) {
        throw new Error(
          `refusing to start: ${root} has uncommitted changes. Commit or stash first, then /task-start ${slug}.`
        )
      }
      const base = gitOk(root, ["rev-parse", "--abbrev-ref", "HEAD"])
      if (!base) throw new Error("could not determine current branch (not a git repo?)")
      if (gitOk(root, ["rev-parse", "--verify", branch])) {
        throw new Error(`branch ${branch} already exists. Pick another slug or merge/delete it first.`)
      }
      if (fs.existsSync(dir)) {
        throw new Error(`worktree directory ${dir} already exists. Pick another slug or remove it first.`)
      }
      try {
        runGit(root, ["worktree", "add", dir, "-b", branch, base])
      } catch (e) {
        throw new Error(`git worktree add failed: ${e}`)
      }
      const record: TaskRecord = { base, branch, directory: dir, createdAt: new Date().toISOString() }
      await ctx.storage.set(taskKey(slug), record)
      try {
        await ctx.session.move({ sessionID, directory: dir })
      } catch (e) {
        console.error(`[task-branch] session.move failed (continue manually): ${e}`)
      }
      return `started task ${slug}\nbase: ${base}\nbranch: ${branch}\ndirectory: ${dir}\nsession moved to worktree (if move failed, continue manually in ${dir}). Make all edits there; ${base} stays clean.`
    }

    async function doDiff(slug: string): Promise<string> {
      validateSlug(slug)
      const stored = (await ctx.storage.get(taskKey(slug))) as TaskRecord | undefined
      const branch = stored?.branch ?? branchName(slug)
      const base = stored?.base ?? gitOk(root, ["merge-base", "HEAD", branch])
      if (!base) throw new Error(`no stored base for ${slug} and merge-base failed. Is ${branch} present?`)
      if (!gitOk(root, ["rev-parse", "--verify", branch])) {
        throw new Error(`branch ${branch} does not exist.`)
      }
      const mergeBase = gitOk(root, ["merge-base", base, branch]) ?? base
      const stat = gitOk(root, ["diff", `${base}...${branch}`, "--stat"]) || "(empty diff)"
      const namesRaw = gitOk(root, ["diff", `${base}...${branch}`, "--name-status"]) || "(no files)"
      const commits = gitOk(root, ["log", "--oneline", `${base}..${branch}`]) || "(no commits — uncommitted work in worktree?)"
      const worktreeStatus = stored ? gitOk(stored.directory, ["status", "--porcelain"]) ?? "" : ""
      const dirtyNote = worktreeStatus.trim()
        ? `\n\nWARNING: uncommitted changes in worktree ${stored?.directory}:\n${worktreeStatus}\nCommit them first — branch diff misses uncommitted work.`
        : ""
      const wtDir = stored?.directory ?? findWorktreeDir(root, branch)
      const visual = buildVisualSection(parseNameStatus(namesRaw), wtDir, base, branch)
      return `diff ${base}...${branch} (merge-base ${mergeBase.slice(0, 8)})\n\nCommits (${base}..${branch}):\n${commits}\n\nStat:\n${stat}\n\nFiles:\n${namesRaw}${dirtyNote}${visual}`
    }

    async function doMerge(slug: string, squash: boolean): Promise<string> {
      validateSlug(slug)
      const stored = (await ctx.storage.get(taskKey(slug))) as TaskRecord | undefined
      if (!stored) throw new Error(`no task record for ${slug}. Did you start it with /task-start ${slug}?`)
      const { base, branch, directory } = stored
      if (!gitOk(root, ["rev-parse", "--verify", branch])) {
        throw new Error(`branch ${branch} does not exist; nothing to merge.`)
      }
      const wtStatus = gitOk(directory, ["status", "--porcelain"]) ?? ""
      if (wtStatus.trim()) {
        throw new Error(`worktree ${directory} has uncommitted changes:\n${wtStatus}\nCommit them first, then /task-merge ${slug}.`)
      }
      const rootStatus = gitOk(root, ["status", "--porcelain"]) ?? ""
      if (rootStatus.trim()) {
        throw new Error(`canonical checkout ${root} is dirty. Commit or stash first, then /task-merge ${slug}.`)
      }
      try {
        runGit(root, ["checkout", base])
      } catch (e) {
        throw new Error(`checkout ${base} failed: ${e}`)
      }
      try {
        if (squash) runGit(root, ["merge", "--squash", branch])
        else runGit(root, ["merge", "--no-ff", branch])
      } catch (e) {
        throw new Error(`merge ${branch} into ${base} failed (resolve conflicts, then commit): ${e}`)
      }
      if (squash) {
        return `squashed ${branch} into ${base} (staged, NOT committed). Review with git status/diff --cached in ${root}, then commit. Cleanup with: git worktree remove ${directory} && git branch -d ${branch}. Storage record kept until you commit — re-run /task-merge ${slug} --cleanup after committing to finish cleanup.`
      }
      try {
        runGit(root, ["worktree", "remove", directory])
      } catch (e) {
        console.error(`[task-branch] worktree remove failed: ${e}`)
      }
      try {
        runGit(root, ["branch", "-d", branch])
      } catch {
        // leave for manual cleanup if not fully merged
      }
      await ctx.storage.remove(taskKey(slug))
      return `merged ${branch} into ${base} (--no-ff). Worktree removed, branch deleted. Verify with git log --oneline -3 in ${root}.`
    }

    async function doCleanup(slug: string): Promise<string> {
      validateSlug(slug)
      const stored = (await ctx.storage.get(taskKey(slug))) as TaskRecord | undefined
      if (!stored) return `no task record for ${slug}; nothing to clean.`
      const { branch, directory } = stored
      try {
        runGit(root, ["worktree", "remove", "--force", directory])
      } catch (e) {
        console.error(`[task-branch] worktree remove --force failed: ${e}`)
      }
      try {
        runGit(root, ["branch", "-D", branch])
      } catch (e) {
        console.error(`[task-branch] branch -D failed: ${e}`)
      }
      await ctx.storage.remove(taskKey(slug))
      return `cleaned up task ${slug}: worktree ${directory} removed, branch ${branch} deleted (forced).`
    }

    async function doList(): Promise<string> {
      const tasks: TaskLiveInfo[] = []
      let after: string | undefined
      do {
        const page = await ctx.storage.scan({ prefix: "task:", after, limit: 100 })
        for (const e of page.entries) {
          const slug = e.key.slice("task:".length)
          const rec = e.value as unknown as TaskRecord
          const branchExists = !!gitOk(root, ["rev-parse", "--verify", rec.branch])
          const worktreeExists = fs.existsSync(rec.directory)
          const wtStatus = worktreeExists ? (gitOk(rec.directory, ["status", "--porcelain"]) ?? "").trim() : ""
          const log = branchExists ? (gitOk(root, ["log", "--oneline", `${rec.base}..${rec.branch}`]) ?? "") : ""
          const ahead = log.trim() ? log.trim().split("\n").length : 0
          tasks.push({ slug, ...rec, branchExists, worktreeExists, dirty: !!wtStatus, ahead })
        }
        after = page.next
      } while (after)
      if (!tasks.length) return "no active tasks (storage empty). Start one with /task-start <slug>."
      return `active tasks (${tasks.length}):\n` + tasks.map(formatTaskRow).join("\n")
    }

    await ctx.tool.transform((editor) => {
      editor.namespace({ name: "task", description: "Task-branch worktree workflow" })
      editor.add({
        name: "start",
        description: "Start a task: capture base branch, create worktree + branch, move session there",
        input: {
          type: "object",
          properties: { slug: { type: "string" } },
          required: ["slug"],
          additionalProperties: false,
        },
        options: { namespace: "task", codemode: true },
        execute: async (input, context) => {
          const { slug } = input as { slug: string }
          try {
            const out = await doStart(context.sessionID, slug)
            return { content: out }
          } catch (e) {
            return { content: `task_start failed: ${(e as Error).message}` }
          }
        },
      })
      editor.add({
        name: "diff",
        description: "Show diff of task branch vs its captured base branch",
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
        description: "List all active task slugs with branch/worktree status",
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
        description: "Start task <slug>: worktree + branch from current branch, move session there",
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
            const out = await doStart(sessionID, slug)
            await ctx.session.prompt({ ...prompt, sessionID, delivery, text: `${out}\n\nProceed with the task in the worktree.` })
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
        description: "Merge task <slug> into its base branch (squash default; --no-squash for merge commit; --cleanup to force-remove worktree)",
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
        description: "List all active task slugs with branch/worktree status",
        execute: async ({ sessionID, prompt, delivery }) => {
          try {
            const out = await doList()
            await ctx.session.prompt({ ...prompt, sessionID, delivery, text: out })
          } catch (e) {
            await ctx.session.prompt({ ...prompt, sessionID, delivery, text: `task-list failed: ${(e as Error).message}` })
          }
        },
      })
    })
  },
})
