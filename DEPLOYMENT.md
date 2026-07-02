# Getting this demo live: 0 → production checklist

Assumes an existing S3 bucket + CloudFront distribution serving your site. The demo
deploys as a subdirectory-style app or its own bucket/distribution — steps below cover
the common "own path on my existing site" and "own subdomain" options where they differ.

## 1. GitHub repo

- [ ] `git init` in `adversarial-playground/`, initial commit (model artifacts in
      `public/model/` are checked in on purpose — they're 24 KB)
- [ ] Create the GitHub repo: `gh repo create adversarial-playground --public --source . --push`
      (public = it's a portfolio piece)
- [ ] In the repo settings, add the deploy configuration (Settings → Secrets and
      variables → Actions):
  - Variable `AWS_REGION` — your bucket's region
  - Variable `S3_BUCKET` — your bucket name
  - Variable `CLOUDFRONT_DISTRIBUTION_ID` — from the CloudFront console
  - Secret `AWS_ROLE_ARN` — created in step 2

## 2. AWS: OIDC role for GitHub Actions (one-time, no long-lived keys)

- [ ] IAM → Identity providers → Add provider: `token.actions.githubusercontent.com`
      (OpenID Connect, audience `sts.amazonaws.com`) — skip if your account already has it
- [ ] Create an IAM role with a trust policy for that provider, with the condition
      `token.actions.githubusercontent.com:sub` = `repo:<you>/adversarial-playground:ref:refs/heads/main`
- [ ] Attach a minimal permissions policy to the role:
  - `s3:PutObject`, `s3:DeleteObject`, `s3:ListBucket` on the bucket + its objects
  - `cloudfront:CreateInvalidation` on the distribution
- [ ] Copy the role ARN into the `AWS_ROLE_ARN` repo secret

## 3. Decide where the demo lives

**Option A — path on your existing site** (e.g. `zack.dev/adversarial-playground/`):
- [ ] Set `base: '/adversarial-playground/'` in `vite.config.ts`
- [ ] Change the two `aws s3 sync` destinations in `.github/workflows/deploy.yml` to
      `s3://$BUCKET/adversarial-playground/...` and prefix the invalidation paths
- [ ] Nothing else changes — the app uses `import.meta.env.BASE_URL` for the model fetch

**Option B — own subdomain** (e.g. `playground.zack.dev`): new bucket + distribution
(or an additional CloudFront behavior), ACM cert covers the subdomain, DNS record in
Route 53. More setup, cleaner URL.

## 4. First deploy

- [ ] Push to `main` (or run the workflow manually — it has `workflow_dispatch`)
- [ ] Watch Actions run: build → sync → invalidation
- [ ] Load the live URL, draw a digit, confirm prediction (open devtools → Network:
      `model.json` + `weights.bin` should come from your domain, ~24 KB total)

## 5. Nice-to-haves (any order)

- [ ] Replace the default Vite `favicon.svg`
- [ ] Add an Open Graph image + meta tags for link previews
- [ ] Link the demo from your site's landing page with a one-liner
      ("runs a CNN in your browser — try to fool it")
- [ ] Lighthouse pass (should already be green: static, no fonts, one JS bundle)

## Costs

Everything here is in permanent free tiers or rounding-error territory: the bundle is
~290 KB gzipped + 24 KB of model, so even thousands of visits are pennies of CloudFront
egress. GitHub Actions is free for public repos. No inference costs — visitors' browsers
do the compute.
