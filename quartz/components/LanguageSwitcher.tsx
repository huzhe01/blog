import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import style from "./styles/explorer.scss" // Using explorer styles for simplicity or custom style
import { classNames } from "../util/lang"

const LanguageSwitcher: QuartzComponent = ({ fileData, displayClass }: QuartzComponentProps) => {
    const slug = fileData.slug
    if (!slug) return null

    const isEnglish = slug.startsWith("en/") || slug === "en"

    let targetSlug = ""
    let label = ""

    if (isEnglish) {
        // Switch to CN
        targetSlug = slug.replace(/^en\/?/, "")
        if (targetSlug === "") targetSlug = "index" // Handle root
        label = "🇨🇳 CN"
    } else {
        // Switch to EN
        targetSlug = slug === "index" ? "en/index" : `en/${slug}`
        label = "🇺🇸 EN"
    }

    // Clean up index for links (Quartz usually handles this, but let's be safe)
    const targetUrl = targetSlug === "index" ? "/" : `/${targetSlug.replace(/\/index$/, "")}`

    return (
        <div class={classNames(displayClass, "language-switcher")}>
            <a href={targetUrl} style={{ fontWeight: "bold", textDecoration: "none" }}>{label}</a>
        </div>
    )
}

// Basic styles (inline for now or reuse existing)
LanguageSwitcher.css = `
.language-switcher {
  margin-top: 0.5rem;
  margin-bottom: 0.5rem;
}
`

export default (() => LanguageSwitcher) satisfies QuartzComponentConstructor
