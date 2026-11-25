// Daily email trigger: a Cloudflare Worker that starts the GitHub "Daily options email" workflow on time.
//
// GitHub's own schedule can start runs late or drop them; a run started through the API ("Run workflow") starts
// within seconds. This Worker fires at 03:55 UTC (09:25 IST) on weekdays, asks GitHub to run the workflow on main,
// and the job waits until 09:30 IST and sends. The GitHub schedule stays as a backup: whichever run comes second
// sees "already sent today" and stops, so nobody gets two emails.
//
// Settings (Cloudflare dashboard > the Worker > Settings > Variables and secrets):
//   GITHUB_TOKEN  (secret)  fine-grained token: this repository only, permission "Actions: Read and write"
//   RUN_KEY       (secret, optional)  lets you start a run by opening  https://<worker>/run?key=<RUN_KEY>
//   GITHUB_REPO   (text, optional)    default ChandrashekharB91/option-pricing-agent
// Trigger: Settings > Trigger events > Cron: 55 3 * * MON-FRI

const WORKFLOW = "daily-feed.yml";

async function dispatch(env) {
  const repo = env.GITHUB_REPO || "ChandrashekharB91/option-pricing-agent";
  if (!env.GITHUB_TOKEN) return { ok: false, status: 0, detail: "GITHUB_TOKEN is not set" };
  const url = `https://api.github.com/repos/${repo}/actions/workflows/${WORKFLOW}/dispatches`;
  let last = { ok: false, status: 0, detail: "" };
  for (let attempt = 1; attempt <= 3; attempt++) {
    try {
      const r = await fetch(url, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${env.GITHUB_TOKEN}`,
          Accept: "application/vnd.github+json",
          "X-GitHub-Api-Version": "2022-11-28",
          "User-Agent": "nifty-options-daily-trigger",
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ ref: "main" }), // no inputs: the regular send to every subscriber
      });
      last = { ok: r.status === 204, status: r.status, detail: r.status === 204 ? "workflow started" : await r.text() };
      if (last.ok || (r.status >= 400 && r.status < 500 && r.status !== 429)) return last; // done, or a settings error
    } catch (e) {
      last = { ok: false, status: 0, detail: String(e) };
    }
    await new Promise((res) => setTimeout(res, 5000 * attempt));
  }
  return last;
}

export default {
  async scheduled(event, env, ctx) {
    const r = await dispatch(env);
    console.log(`daily email trigger at ${new Date(event.scheduledTime).toISOString()}: ${r.status} ${r.detail}`);
    if (!r.ok) throw new Error(`GitHub dispatch failed: ${r.status} ${r.detail}`); // shows as an error in the logs
  },

  async fetch(request, env) {
    const u = new URL(request.url);
    if (u.pathname === "/run") {
      if (!env.RUN_KEY || u.searchParams.get("key") !== env.RUN_KEY) return new Response("forbidden", { status: 403 });
      const r = await dispatch(env);
      return Response.json(r, { status: r.ok ? 200 : 502 });
    }
    return new Response("NIFTY Options Daily trigger: starts the GitHub email workflow at 09:25 IST on weekdays.\n");
  },
};
