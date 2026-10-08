# =====================================================================================
# Market Basket Analysis in R — built from first principles (base R only)
# Author: Fordrane Albert Okumu
#
# WHY BASE R?
#   The usual R tool is the {arules} package (apriori(), inspect()). This script
#   rebuilds the same metrics with plain matrix algebra instead, so that:
#     1. every number can be traced back to a formula (good for teaching and audit);
#     2. it runs anywhere R runs, with no package installs;
#     3. it independently CROSS-CHECKS the Python / mlxtend results in
#        python/market_basket_analysis.ipynb. Two implementations agreeing is strong
#        evidence that the rules are computed correctly.
#   The {arules} equivalent is shown at the end, for readers who have it installed.
#
# WHAT IT DOES
#   1. Load transactions and build the basket x product binary matrix
#   2. Item support (how popular each product is)
#   3. Pair rules  A -> B         via one matrix product  t(M) %*% M
#   4. Triple rules {A,B} -> C    for every frequent pair
#   5. Support, confidence, lift, leverage, conviction
#   6. Fisher exact test + Bonferroni correction (are rules real or noise?)
#   7. Store-type segment comparison
#   8. Charts + a Markdown results report (outputs/r/R_results.md)
#   9. Cross-check against the Python results
#
# Run from the repository root:   Rscript R/market_basket_analysis.R
# =====================================================================================

MIN_SUPPORT <- 0.01   # itemset must appear in >= 1% of baskets (150 of 15,000)
MIN_CONF    <- 0.30   # at least 30% of A-baskets also contain B
MIN_LIFT    <- 1.20   # B at least 20% more likely when A is present

out_dir <- file.path("outputs", "r")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
TEAL <- "#0f766e"; AMBER <- "#b45309"

# -------------------------------------------------------------------------------------
# 1. Load data and build the binary basket matrix
# -------------------------------------------------------------------------------------
# Raw data has one row per product per invoice ("long" format). Association rules
# need a binary matrix: rows = baskets, columns = products, 1 = product in basket.
# Quantity is deliberately ignored: we care WHETHER items are bought together.
tx <- read.csv("data/transactions.csv", stringsAsFactors = FALSE)
M  <- unclass(table(tx$invoice_id, tx$product)) > 0     # logical matrix
storage.mode(M) <- "integer"                            # 0/1 for fast algebra
n  <- nrow(M)
cat(sprintf("Baskets: %s | products: %d | matrix density: %.1f%%\n",
            format(n, big.mark = ","), ncol(M), 100 * mean(M)))

# -------------------------------------------------------------------------------------
# 2. Item support = share of baskets containing the item
# -------------------------------------------------------------------------------------
item_support <- sort(colMeans(M), decreasing = TRUE)
cat("\nMost popular products (support):\n")
print(round(head(item_support, 8), 3))

# -------------------------------------------------------------------------------------
# 3. Pair counts in one step: C = t(M) %*% M
#    C[i, j] = number of baskets containing BOTH product i and product j.
#    The diagonal C[i, i] is the number of baskets containing product i.
# -------------------------------------------------------------------------------------
C <- crossprod(M)
products <- colnames(M)

make_rule <- function(lhs, rhs, n_lhs, n_rhs, n_both) {
  supp   <- n_both / n
  conf   <- n_both / n_lhs
  supp_r <- n_rhs / n
  lift   <- conf / supp_r
  data.frame(lhs = lhs, rhs = rhs, baskets = n_both, support = supp, confidence = conf,
             lift = lift, leverage = supp - (n_lhs / n) * supp_r,
             conviction = ifelse(conf < 1, (1 - supp_r) / (1 - conf), Inf),
             n_lhs = n_lhs, n_rhs = n_rhs, stringsAsFactors = FALSE)
}

idx  <- which(C >= MIN_SUPPORT * n & row(C) != col(C), arr.ind = TRUE)
pair_rules <- make_rule(products[idx[, 1]], products[idx[, 2]],
                        diag(C)[idx[, 1]], diag(C)[idx[, 2]], C[idx])
cat(sprintf("\nFrequent pairs: %d  -> candidate pair rules (both directions): %d\n",
            nrow(idx) / 2, nrow(pair_rules)))

# -------------------------------------------------------------------------------------
# 4. Triple rules {A, B} -> C
#    Apriori principle: a triple can only be frequent if all its pairs are frequent,
#    so we only extend FREQUENT pairs. For each pair we take the baskets that contain
#    both items and count every other product inside those baskets.
# -------------------------------------------------------------------------------------
freq_pairs <- idx[idx[, 1] < idx[, 2], , drop = FALSE]
triple_list <- lapply(seq_len(nrow(freq_pairs)), function(k) {
  a <- freq_pairs[k, 1]; b <- freq_pairs[k, 2]
  rows <- M[, a] == 1 & M[, b] == 1
  co <- colSums(M[rows, , drop = FALSE])
  co[c(a, b)] <- 0
  keep <- which(co >= MIN_SUPPORT * n)
  if (!length(keep)) return(NULL)
  make_rule(paste(sort(products[c(a, b)]), collapse = " + "), products[keep],
            sum(rows), diag(C)[keep], co[keep])
})
triple_rules <- do.call(rbind, triple_list)
rules <- rbind(pair_rules, triple_rules)
rules <- rules[rules$confidence >= MIN_CONF & rules$lift >= MIN_LIFT, ]
rules$rule <- paste(rules$lhs, "->", rules$rhs)
rules <- rules[order(-rules$lift), ]
cat(sprintf("Rules after confidence >= %.2f and lift >= %.2f: %d (%d pair, %d triple)\n",
            MIN_CONF, MIN_LIFT, nrow(rules), sum(!grepl("\\+", rules$lhs)), sum(grepl("\\+", rules$lhs))))

# -------------------------------------------------------------------------------------
# 5. Statistical significance: one-sided Fisher exact test per rule.
#    2x2 table: (LHS present/absent) x (RHS present/absent).
#    Bonferroni: significant only if p < 0.05 / number_of_rules.
# -------------------------------------------------------------------------------------
rules$p_value <- mapply(function(nl, nr, nb) {
  tab <- matrix(c(nb, nl - nb, nr - nb, n - nl - nr + nb), 2, byrow = TRUE)
  fisher.test(tab, alternative = "greater")$p.value
}, rules$n_lhs, rules$n_rhs, rules$baskets)
alpha <- 0.05 / nrow(rules)
rules$significant <- rules$p_value < alpha
cat(sprintf("Significant after Bonferroni (p < %.2e): %d of %d\n", alpha, sum(rules$significant), nrow(rules)))

cat("\nTop 12 rules by lift:\n")
print(transform(head(rules[, c("rule", "baskets", "support", "confidence", "lift", "conviction")], 12),
                support = round(support, 3), confidence = round(confidence, 3),
                lift = round(lift, 2), conviction = round(conviction, 2)), row.names = FALSE)
write.csv(rules[, c("lhs", "rhs", "baskets", "support", "confidence", "lift", "leverage",
                    "conviction", "p_value", "significant")],
          file.path(out_dir, "association_rules_R.csv"), row.names = FALSE)

# -------------------------------------------------------------------------------------
# 6. Segment comparison: supermarket vs duka (pair rules with lift >= 1)
# -------------------------------------------------------------------------------------
store <- tapply(tx$store_type, tx$invoice_id, `[`, 1)[rownames(M)]
segment_pairs <- function(mask) {
  Ms <- M[mask, , drop = FALSE]; ns <- nrow(Ms); Cs <- crossprod(Ms)
  ix <- which(Cs >= MIN_SUPPORT * ns & row(Cs) != col(Cs), arr.ind = TRUE)
  conf <- Cs[ix] / diag(Cs)[ix[, 1]]
  d <- data.frame(rule = paste(products[ix[, 1]], "->", products[ix[, 2]]),
                  confidence = conf, lift = conf / (diag(Cs)[ix[, 2]] / ns))
  d[d$lift >= 1, ]                       # positive associations only, as in the Python notebook
}
seg <- merge(segment_pairs(store == "Supermarket"), segment_pairs(store == "Duka"),
             by = "rule", all = TRUE, suffixes = c("_supermarket", "_duka"))
seg <- seg[order(-seg$confidence_duka), ]
write.csv(seg, file.path(out_dir, "segment_rules_R.csv"), row.names = FALSE)
cat(sprintf("\nPair rules - supermarket: %d, duka: %d, supermarket-only: %d\n",
            sum(!is.na(seg$lift_supermarket)), sum(!is.na(seg$lift_duka)), sum(is.na(seg$lift_duka))))

# -------------------------------------------------------------------------------------
# 7. Charts
# -------------------------------------------------------------------------------------
png(file.path(out_dir, "R_01_item_frequency.png"), width = 1600, height = 1000, res = 170)
par(mar = c(4, 12, 3, 2))
top_items <- rev(head(item_support, 15))
barplot(100 * top_items, horiz = TRUE, las = 1, col = TEAL, border = NA, cex.names = .8,
        xlab = "% of baskets", main = "Top 15 products by support (R)")
invisible(dev.off())

png(file.path(out_dir, "R_02_top_rules.png"), width = 1800, height = 1100, res = 170)
par(mar = c(4, 22, 3, 2))
tr <- head(rules[rules$significant, ], 15)[15:1, ]
barplot(tr$lift, names.arg = tr$rule, horiz = TRUE, las = 1, cex.names = .7, border = NA,
        col = ifelse(grepl("\\+", tr$lhs), AMBER, TEAL),
        xlab = "Lift", main = "Top 15 significant rules by lift (R)")
invisible(dev.off())

# Lift heatmap for the 15 most popular products: red = bought together more than chance
top15 <- names(head(item_support, 15))
L <- (C[top15, top15] / n) / outer(item_support[top15], item_support[top15])
diag(L) <- NA
png(file.path(out_dir, "R_03_lift_heatmap.png"), width = 1700, height = 1500, res = 170)
par(mar = c(10, 10, 3, 2))
image(1:15, 1:15, log2(L), col = hcl.colors(21, "Blue-Red 3"), zlim = c(-2.5, 2.5),
      axes = FALSE, xlab = "", ylab = "", main = "Pairwise lift, top 15 products (log2 scale; red = together)")
axis(1, 1:15, top15, las = 2, cex.axis = .7); axis(2, 1:15, top15, las = 1, cex.axis = .7)
for (i in 1:15) for (j in 1:15) if (!is.na(L[i, j])) text(i, j, sprintf("%.1f", L[i, j]), cex = .55)
invisible(dev.off())

# -------------------------------------------------------------------------------------
# 8. Cross-check against the Python (mlxtend) output
#    The Python notebook saves its pruned rule list. Every one of those rules with a
#    1- or 2-item left side must exist here with the same metrics.
# -------------------------------------------------------------------------------------
py_file <- file.path("outputs", "python", "association_rules.csv")
check_txt <- "Python results not found - run the notebook first to enable the cross-check."
if (file.exists(py_file)) {
  py <- read.csv(py_file, stringsAsFactors = FALSE)
  py$rule <- sub("  ->  ", " -> ", py$rule)
  py <- py[lengths(regmatches(py$rule, gregexpr("\\+", sub(" ->.*", "", py$rule)))) <= 1, ]
  m <- merge(py, rules, by = "rule", suffixes = c("_py", "_r"))
  max_diff <- max(abs(c(m$support_py - m$support_r, m$confidence_py - m$confidence_r,
                        m$lift_py - m$lift_r)))
  check_txt <- sprintf("Python rules checked: %d | matched in R: %d | max absolute metric difference: %.2e",
                       nrow(py), nrow(m), max_diff)
}
cat("\nCROSS-CHECK:", check_txt, "\n")

# -------------------------------------------------------------------------------------
# 9. Write a GitHub-readable Markdown report of the R results
# -------------------------------------------------------------------------------------
md_table <- function(df) {
  c(paste("|", paste(names(df), collapse = " | "), "|"),
    paste("|", paste(rep("---", ncol(df)), collapse = " | "), "|"),
    apply(df, 1, function(r) paste("|", paste(r, collapse = " | "), "|")))
}
fmt_rules <- function(d) data.frame(rule = gsub("->", "→", d$rule), baskets = d$baskets,
                                    support = sprintf("%.3f", d$support),
                                    confidence = sprintf("%.0f%%", 100 * d$confidence),
                                    lift = sprintf("%.2f", d$lift))
top_seg <- head(seg[!is.na(seg$confidence_duka) & !is.na(seg$confidence_supermarket), ], 6)
report <- c(
  "# Market Basket Analysis — R results (auto-generated by `R/market_basket_analysis.R`)", "",
  sprintf("*%s baskets · %d products · min support %.0f%% · min confidence %.0f%% · min lift %.1f*",
          format(n, big.mark = ","), ncol(M), 100 * MIN_SUPPORT, 100 * MIN_CONF, MIN_LIFT), "",
  "## Rule counts", "",
  sprintf("- Rules passing thresholds: **%d** (%d pair rules, %d triple rules)",
          nrow(rules), sum(!grepl("\\+", rules$lhs)), sum(grepl("\\+", rules$lhs))),
  sprintf("- Significant after Fisher exact test + Bonferroni: **%d**", sum(rules$significant)),
  sprintf("- **Cross-check with Python:** %s", check_txt), "",
  "## Top 10 rules by lift", "", md_table(fmt_rules(head(rules, 10))), "",
  "## Highest-support rules (the everyday missions)", "",
  md_table(fmt_rules(head(rules[order(-rules$support), ], 8))), "",
  "## Channel comparison — highest-confidence rules in dukas", "",
  md_table(data.frame(rule = gsub("->", "→", top_seg$rule),
                      `conf supermarket` = sprintf("%.0f%%", 100 * top_seg$confidence_supermarket),
                      `conf duka` = sprintf("%.0f%%", 100 * top_seg$confidence_duka),
                      `lift supermarket` = sprintf("%.2f", top_seg$lift_supermarket),
                      `lift duka` = sprintf("%.2f", top_seg$lift_duka), check.names = FALSE)), "",
  "## Charts", "",
  "![Item frequency](R_01_item_frequency.png)", "", "![Top rules](R_02_top_rules.png)", "",
  "![Lift heatmap](R_03_lift_heatmap.png)", "",
  "**Reading the heatmap:** each cell is the lift between two popular products. Values above 1 (red) mean the pair is bought together more often than chance would predict. Values below 1 (blue) mean they are bought together *less* often than chance, which usually signals different shopping missions. Here the breakfast/tea items (milk, bread, tea, sugar) score about 0.6-0.7 against the dinner-cook items (maize flour, oil, onions, tomatoes, rice): people tend to buy for one mission per trip.")
writeLines(report, file.path(out_dir, "R_results.md"))
cat("Wrote outputs/r/R_results.md, CSVs and charts\n")

# -------------------------------------------------------------------------------------
# Equivalent with the {arules} package (for reference; not required to run this script)
# -------------------------------------------------------------------------------------
# library(arules)
# trans <- as(split(tx$product, tx$invoice_id), "transactions")
# r <- apriori(trans, parameter = list(supp = 0.01, conf = 0.30, minlen = 2, maxlen = 3))
# r <- subset(r, lift >= 1.2)
# inspect(head(sort(r, by = "lift"), 10))
# is.significant(r, trans, method = "fisher", adjust = "bonferroni")
