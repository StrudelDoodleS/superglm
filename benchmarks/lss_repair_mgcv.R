args <- commandArgs(trailingOnly=TRUE)
usage <- paste("Usage: Rscript benchmarks/lss_repair_mgcv.R",
               "DATA.csv OUTPUT_DIR [default|unbounded] [train|full]")
if (length(args) == 1 && args[1] %in% c("-h", "--help")) {
    writeLines(usage)
    quit(status=0)
}
if (length(args) < 2 || length(args) > 4) stop(usage)
mode <- if (length(args) >= 3) args[3] else "default"
split <- if (length(args) >= 4) args[4] else "train"
if (!mode %in% c("default", "unbounded")) stop(usage)
if (!split %in% c("train", "full")) stop(usage)
data_path <- normalizePath(args[1], mustWork=TRUE)
out <- args[2]
full <- split == "full"
label <- if (full) paste0(mode, "_full") else mode
suppressPackageStartupMessages(library(mgcv))
d <- read.csv(data_path)
if (!dir.exists(out) && !dir.create(out, recursive=TRUE)) {
    stop("Cannot create output directory: ", out)
}
d <- if (full) d[d$split != "test", ] else d[d$split == "train", ]
cats <- c("Area", "VehBrand", "VehGas", "Region")
nums <- c("VehAge", "DrivAge", "BonusMalus", "LogDensity", "VehPower")
for (x in cats) d[[x]] <- factor(d[[x]])
terms <- c(sprintf("s(%s, bs='cr', k=8)", nums), cats)
mean_formula <- as.formula(paste("y ~", paste(terms, collapse=" + ")))
scale_formula <- as.formula(paste("~", paste(terms, collapse=" + ")))
family <- if (mode == "unbounded") gammals(link=list("identity", "identity")) else gammals()
knots <- setNames(lapply(nums, function(x) seq(min(d[[x]]), max(d[[x]]), length.out=8)), nums)
warnings_seen <- character()
start <- proc.time()[["elapsed"]]
fit <- withCallingHandlers(
    gam(list(mean_formula, scale_formula), data=d, family=family,
        knots=knots, method="REML", control=gam.control(nthreads=1)),
    warning=function(w) {
        warnings_seen <<- unique(c(warnings_seen, conditionMessage(w)))
    }
)
elapsed <- proc.time()[["elapsed"]] - start
pred <- predict(fit, type="response")
mu <- pred[, 1]
cv <- exp(pred[, 2] / 2)
receipt <- list(mode=mode, split=split, data_path=data_path,
                data_md5=unname(tools::md5sum(data_path)),
                n_rows=nrow(d), mgcv=as.character(packageVersion("mgcv")),
                elapsed_seconds=elapsed, coefficient_converged=fit$converged,
                outer=fit$outer.info, minimum_cv=min(cv), maximum_cv=max(cv),
                mean_range=range(mu), nll=-mean(dgamma(d$y, shape=1/cv^2,
                    scale=mu*cv^2, log=TRUE)), smoothing=fit$sp,
                warnings=warnings_seen, coef_count=length(coef(fit)))
dput(receipt, file=file.path(out, paste0("mgcv_", label, "_receipt.R")))
saveRDS(fit, file=file.path(out, paste0("mgcv_", label, ".rds")))
write.csv(data.frame(row_id=d$row_id, mu=mu, cv=cv),
          file.path(out, paste0("mgcv_", label, "_predictions.csv")), row.names=FALSE)
print(receipt)
