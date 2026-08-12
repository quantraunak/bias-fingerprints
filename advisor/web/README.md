# Marula — website

Next.js 16 · React 19 · Tailwind v4. Static, no backend, deploys to Vercel as-is.

```bash
npm install
npm run dev      # localhost:3000
npm run build    # verify before deploying
```

## Deploy

```bash
npx vercel        # preview
npx vercel --prod # production
```

Set the root directory to `web/` if you deploy the whole `marula` repo from the
Vercel dashboard.

## Routes

| Route | Purpose |
|---|---|
| `/` | Positioning, the four mandates, research figures, who it's for |
| `/strategy` | The five decisions, the long/short extension, what we won't claim |
| `/access` | Intake form and engagement terms |

## Design system

Tokens live in `app/globals.css` under `@theme` — change them there, not in
components.

- **Paper `#f6f6f4` / ink `#12161f`** — cool document ground, not a fintech app
- **Navy `#16283d`** — primary accent, carries authority
- **Brass `#8a6d33`** — reserved for figures and emphasis; never decorative
- **Newsreader** display serif · **Geist** UI · **Geist Mono** for all data

Figures use `.tnum` (tabular numerals) so columns align.

## Before this goes to anyone

1. **Wire the form.** It currently opens a mail client via `mailto:`. Replace with
   a real endpoint (Formspree, Resend, or a route handler). **Never collect
   account statements or cost-basis data over unencrypted email.**
2. **Replace the contact details** — `hello@marula.example` is a placeholder.
3. **Check the name.** "Marula" is used by several financial firms. Run a
   trademark and entity search before printing it on anything.
4. **Have securities counsel read every page.** Specifically:
   - Performance figures on `/` are **simulated backtest results**, labeled as
     such. SEC Marketing Rule 206(4)-1 governs how hypothetical performance may
     be presented and to whom. This is the single highest-risk element on the
     site.
   - The footer states Marula is not a registered adviser. That must stay
     accurate, and the moment you provide personalized advice for compensation,
     registration is likely required.
5. **Verify the backtest figures resolve** to a current run of
   `r1000-ls-strategy` — they are hardcoded in `app/page.tsx` (`RESEARCH`).
