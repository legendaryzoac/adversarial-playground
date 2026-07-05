/**
 * Satellite-site nav matching zackwithers.com — same tokens and layout as the
 * blog's "brand / section" header, with an outlined back-link to the main site.
 */
export default function SiteNav() {
  return (
    <nav className="bg-site/85 border-line fixed inset-x-0 top-0 z-50 flex h-[60px] items-center justify-between border-b px-5 backdrop-blur-md sm:px-10">
      <div className="flex items-baseline gap-2.5">
        <a
          href="https://zackwithers.com"
          className="text-accent font-mono text-sm tracking-wider whitespace-nowrap"
        >
          zw ~
        </a>
        <span className="text-muted hidden font-mono text-xs min-[480px]:inline">/</span>
        <span className="text-accent hidden font-mono text-xs whitespace-nowrap min-[480px]:inline">
          adversarial-playground
        </span>
      </div>
      <div className="flex items-center gap-6">
        <a
          href="https://interp.zackwithers.com"
          target="_blank"
          rel="noopener"
          className="font-display text-muted hover:text-fg hidden text-sm font-medium transition-colors sm:block"
        >
          Interp ↗
        </a>
        <a
          href="https://github.com/legendaryzoac/adversarial-playground"
          target="_blank"
          rel="noopener"
          className="font-display text-muted hover:text-fg hidden text-sm font-medium transition-colors sm:block"
        >
          GitHub ↗
        </a>
        <a
          href="https://zackwithers.com"
          className="border-accent text-accent font-display hover:bg-accent hover:text-site rounded border px-3.5 py-1.5 text-[0.82rem] font-medium whitespace-nowrap transition-colors"
        >
          ← zackwithers.com
        </a>
      </div>
    </nav>
  )
}
