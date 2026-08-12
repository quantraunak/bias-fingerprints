import type { Metadata } from "next";
import { Newsreader, Geist, Geist_Mono } from "next/font/google";
import Link from "next/link";
import "./globals.css";

const newsreader = Newsreader({
  subsets: ["latin"],
  variable: "--font-newsreader",
  weight: ["300", "400", "500"],
  style: ["normal", "italic"],
  display: "swap",
});

const geist = Geist({
  subsets: ["latin"],
  variable: "--font-geist-sans",
  display: "swap",
});

const geistMono = Geist_Mono({
  subsets: ["latin"],
  variable: "--font-geist-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Marula — Investment management for families",
  description:
    "We manage a family's whole balance sheet as one portfolio, after tax — not one account at a time.",
};

const NAV = [
  { href: "/tool", label: "Review your portfolio" },
  { href: "/strategy", label: "How it works" },
];

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body
        className={`${newsreader.variable} ${geist.variable} ${geistMono.variable}`}
      >
        <header className="sticky top-0 z-50 border-b border-rule bg-paper/85 backdrop-blur-md">
          <div className="mx-auto flex h-16 max-w-6xl items-center justify-between px-6">
            <Link
              href="/"
              className="font-mono text-[0.78rem] font-medium uppercase tracking-[0.24em] text-ink"
            >
              Marula
            </Link>
            <nav className="flex items-center gap-8">
              {NAV.map((item) => (
                <Link
                  key={item.href}
                  href={item.href}
                  className="font-mono text-[0.72rem] uppercase tracking-[0.14em] text-muted transition-colors hover:text-ink"
                >
                  {item.label}
                </Link>
              ))}
              <Link
                href="/tool"
                className="hidden border border-veld px-4 py-2 font-mono text-[0.7rem] uppercase tracking-[0.14em] text-veld transition-colors hover:bg-veld hover:text-white sm:block"
              >
                Open the tool
              </Link>
            </nav>
          </div>
        </header>

        <main>{children}</main>

        <footer className="mt-32 border-t border-rule bg-sunk">
          <div className="mx-auto max-w-6xl px-6 py-16">
            <div className="grid gap-12 md:grid-cols-[1.4fr_1fr_1fr]">
              <div>
                <div className="font-mono text-[0.78rem] font-medium uppercase tracking-[0.24em] text-ink">
                  Marula
                </div>
                <p className="mt-4 max-w-sm font-display text-lg leading-relaxed text-ink-2">
                  A free tool that reviews a family's whole balance sheet
                  after tax — every account at once, not one at a time.
                </p>
              </div>
              <div>
                <div className="eyebrow">Firm</div>
                <ul className="mt-4 space-y-2.5">
                  {NAV.map((item) => (
                    <li key={item.href}>
                      <Link
                        href={item.href}
                        className="text-sm text-ink-2 transition-colors hover:text-veld"
                      >
                        {item.label}
                      </Link>
                    </li>
                  ))}
                </ul>
              </div>
              <div>
                <div className="eyebrow">Contact</div>
                <ul className="mt-4 space-y-2.5 text-sm text-ink-2">
                  <li>
                    <a
                      href="mailto:raunak.sood@gmail.com"
                      className="transition-colors hover:text-veld"
                    >
                      raunak.sood@gmail.com
                    </a>
                  </li>
                  <li>Los Altos, California</li>
                </ul>
              </div>
            </div>

            <div className="mt-14 space-y-3 border-t border-rule pt-8 font-mono text-[0.68rem] leading-relaxed text-muted">
              <p>
                Marula is a free analysis tool. We are not a registered
                investment adviser, we are not paid for anything on this site,
                and we do not manage money or hold client assets. Nothing here
                is investment, tax, or legal advice, or a recommendation to buy
                or sell any security. The tool applies general rules to figures
                you enter yourself and cannot know your full circumstances.
              </p>
              <p>
                Figures described as research or simulated are model output
                under stated assumptions. They are derived with hindsight, do
                not reflect live trading or actual client accounts, and do not
                indicate future results. Simulated results have inherent
                limitations and typically differ materially from realized
                performance.
              </p>
              <p>© {new Date().getFullYear()} Marula. All rights reserved.</p>
            </div>
          </div>
        </footer>
      </body>
    </html>
  );
}
