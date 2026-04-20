SOURCE_LABELS = {
    "collaborative": "相似用户推荐",
    "content": "风格匹配推荐",
    "hybrid": "综合推荐",
}


class Explainer:
    """Generate human-readable explanations for recommendations."""

    def generate_text(self, rec: dict) -> str:
        """Generate a natural language explanation for a single recommendation."""
        title = rec.get("title", "")
        source = rec.get("source", "")
        source_label = SOURCE_LABELS.get(source, source)
        reason = rec.get("reason", "")

        parts = [f"[{source_label}] {title}: {reason}"]

        if rec.get("cf_score", 0) > 0 and rec.get("cb_score", 0) > 0:
            parts.append(
                f"协同过滤得分 {rec['cf_score']:.2f}, 内容相似度 {rec['cb_score']:.2f}"
            )
        elif rec.get("cf_score", 0) > 0:
            parts.append(f"协同过滤得分 {rec['cf_score']:.2f}")
        elif rec.get("cb_score", 0) > 0:
            parts.append(f"内容相似度 {rec['cb_score']:.2f}")

        return " | ".join(parts)

    def generate_source_label(self, source: str) -> str:
        """Return a short Chinese label for the recommendation source."""
        return SOURCE_LABELS.get(source, source)
