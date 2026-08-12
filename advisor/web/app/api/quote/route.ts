import { NextResponse } from "next/server";

/**
 * Live quotes, proxied server-side.
 *
 * Only TICKERS cross the network — never share counts, cost basis, or account
 * balances. Those stay in the browser, which is the whole privacy model. The
 * proxy exists so the client is not blocked by CORS, not so we can see
 * anything.
 *
 * Source is Yahoo's public chart endpoint — free and keyless. It is an
 * undocumented API, so it can change without notice; a production build should
 * sit on a paid feed with an SLA rather than this.
 */

export const runtime = "nodejs";
export const revalidate = 0;

interface Quote {
  ticker: string;
  price: number | null;
  asOf: string | null;
  error?: string;
}

async function fetchOne(ticker: string): Promise<Quote> {
  const url =
    `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(ticker)}` +
    `?range=1d&interval=1d`;

  try {
    const res = await fetch(url, {
      cache: "no-store",
      // The endpoint rejects requests without a browser-like agent.
      headers: { "User-Agent": "Mozilla/5.0" },
      signal: AbortSignal.timeout(6000),
    });
    if (!res.ok) {
      return { ticker, price: null, asOf: null, error: `upstream ${res.status}` };
    }

    const data = await res.json();
    const meta = data?.chart?.result?.[0]?.meta;
    const price = Number(meta?.regularMarketPrice);
    if (!Number.isFinite(price) || price <= 0) {
      return { ticker, price: null, asOf: null, error: "not found" };
    }

    const ts = Number(meta?.regularMarketTime);
    const asOf = Number.isFinite(ts)
      ? new Date(ts * 1000).toISOString().slice(0, 10)
      : null;

    return { ticker, price, asOf };
  } catch {
    return { ticker, price: null, asOf: null, error: "unreachable" };
  }
}

export async function POST(request: Request) {
  let tickers: string[] = [];
  try {
    const body = await request.json();
    tickers = Array.isArray(body?.tickers) ? body.tickers : [];
  } catch {
    return NextResponse.json({ error: "Expected { tickers: string[] }" }, { status: 400 });
  }

  const clean = Array.from(
    new Set(
      tickers
        .filter((t): t is string => typeof t === "string")
        .map((t) => t.trim().toUpperCase())
        .filter((t) => t.length > 0 && t.length <= 12 && /^[A-Z0-9.\-]+$/.test(t)),
    ),
  ).slice(0, 40);

  if (clean.length === 0) return NextResponse.json({ quotes: [] });

  const quotes = await Promise.all(clean.map(fetchOne));
  return NextResponse.json({ quotes });
}
