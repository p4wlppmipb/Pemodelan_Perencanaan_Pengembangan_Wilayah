# =========================================================
# MODEL MAKSIMISASI ENTROPI KLASIK - AUGMENTED DOUBLY CONSTRAINT 
# POISSON GLM + SIGMA-RESTRICTED PARAMETERIZATION
# DISTANCE DIRECT + DELTA METHOD + MODEL FIT
# SINGLE INTEGRATED CSV OUTPUT
# BUKU PEMODELAN PERENCANAAN PENGEMBANGAN WILAYAH
# =========================================================

install.packages("readr")
install.packages("dplyr")

library(readr)
library(dplyr)

# =========================================================
# 1. READ DATA
# =========================================================

df <- read_csv("Jabodetabek_entropi_2011.csv")

# Required columns:
# origin, destination, flow, distance

df <- df %>%
  filter(
    !is.na(origin),
    !is.na(destination),
    !is.na(flow),
    !is.na(distance),
    flow >= 0,
    distance > 0
  )

# =========================================================
# 2. KEEP ORIGINAL ORDER FROM DATA
# =========================================================

origin_order <- unique(df$origin)
destination_order <- unique(df$destination)

df <- df %>%
  mutate(
    origin = factor(origin, levels = origin_order),
    destination = factor(destination, levels = origin_order),
    mean_pdrb =
      (pdrb_origin + pdrb_destination) / 2,
    
    sd_pdrb =
      sqrt(
        (
          (pdrb_origin - mean_pdrb)^2 +
            (pdrb_destination - mean_pdrb)^2
        ) / 2
      ),
    
    cv_pdrb =
      sd_pdrb / mean_pdrb  
    )

# =========================================================
# 3. SIGMA-RESTRICTED CONTRAST
# =========================================================

contrasts(df$origin) <- contr.sum(nlevels(df$origin))
contrasts(df$destination) <- contr.sum(nlevels(df$destination))

# =========================================================
# 4. ESTIMATE POISSON GLM
# =========================================================

model <- glm(
  flow ~ origin + destination + distance + cv_pdrb,
  family = poisson(link = "log"),
  data = df
)

model_summary <- summary(model)
coef_table <- as.data.frame(model_summary$coefficients)
vc <- vcov(model)

# =========================================================
# 5. MODEL FIT STATISTICS
# =========================================================

pseudo_r2_mcfadden <- 1 - (as.numeric(logLik(model)) / as.numeric(logLik(update(model, . ~ 1))))

deviance_explained <- 1 - (model$deviance / model$null.deviance)

model_aic <- AIC(model)
model_bic <- BIC(model)
model_loglik <- as.numeric(logLik(model))

model_fit_df <- data.frame(
  parameter_type = "model_fit",
  region = c(
    "Pseudo_R2_McFadden",
    "Deviance_Explained",
    "AIC",
    "BIC",
    "LogLikelihood"
  ),
  coefficient = c(
    pseudo_r2_mcfadden,
    deviance_explained,
    model_aic,
    model_bic,
    model_loglik
  ),
  std_error = NA,
  z_value = NA,
  wald_statistic = NA,
  p_value = NA,
  multiplicative_factor = NA
)

# =========================================================
# 6. FUNCTION: FULL SIGMA-RESTRICTED COEFFICIENTS
# =========================================================

get_sigma_effects <- function(model, vc, factor_name, factor_levels) {
  
  coef_names <- names(coef(model))
  effect_names <- grep(paste0("^", factor_name), coef_names, value = TRUE)
  
  k <- length(factor_levels)
  m <- length(effect_names)
  
  beta_est <- coef(model)[effect_names]
  
  beta_full <- c(beta_est, -sum(beta_est))
  names(beta_full) <- factor_levels
  
  vc_sub <- vc[effect_names, effect_names, drop = FALSE]
  
  se_full <- numeric(k)
  se_full[1:m] <- sqrt(diag(vc_sub))
  se_full[k] <- sqrt(sum(vc_sub))
  
  z_full <- beta_full / se_full
  wald_full <- z_full^2
  p_full <- 2 * (1 - pnorm(abs(z_full)))
  
  data.frame(
    parameter_type = factor_name,
    region = factor_levels,
    coefficient = as.numeric(beta_full),
    std_error = as.numeric(se_full),
    z_value = as.numeric(z_full),
    wald_statistic = as.numeric(wald_full),
    p_value = as.numeric(p_full),
    multiplicative_factor = exp(as.numeric(beta_full))
  )
}

# =========================================================
# 7. INTERCEPT
# =========================================================

intercept_df <- data.frame(
  parameter_type = "intercept",
  region = "global_system",
  coefficient = coef_table["(Intercept)", "Estimate"],
  std_error = coef_table["(Intercept)", "Std. Error"],
  z_value = coef_table["(Intercept)", "z value"],
  wald_statistic = coef_table["(Intercept)", "z value"]^2,
  p_value = coef_table["(Intercept)", "Pr(>|z|)"],
  multiplicative_factor = exp(coef_table["(Intercept)", "Estimate"])
)

# =========================================================
# 8. DISTANCE COEFFICIENT
# =========================================================

distance_df <- data.frame(
  parameter_type = "distance",
  region = "distance",
  coefficient = coef_table["distance", "Estimate"],
  std_error = coef_table["distance", "Std. Error"],
  z_value = coef_table["distance", "z value"],
  wald_statistic = coef_table["distance", "z value"]^2,
  p_value = coef_table["distance", "Pr(>|z|)"],
  multiplicative_factor = exp(coef_table["distance", "Estimate"])
)

# =========================================================
# 9. CV COEFFICIENTS
# =========================================================
cv_df <- data.frame(
  parameter_type = "economic_disparity",
  region = "cv_pdrb",
  coefficient = coef_table["cv_pdrb", "Estimate"],
  std_error = coef_table["cv_pdrb", "Std. Error"],
  z_value = coef_table["cv_pdrb", "z value"],
  wald_statistic = coef_table["cv_pdrb", "z value"]^2,
  p_value = coef_table["cv_pdrb", "Pr(>|z|)"],
  multiplicative_factor = exp(coef_table["cv_pdrb", "Estimate"])
)

# =========================================================
# 10. ORIGIN & DESTINATION COEFFICIENTS
# =========================================================

origin_df <- get_sigma_effects(
  model = model,
  vc = vc,
  factor_name = "origin",
  factor_levels = levels(df$origin)
)

destination_df <- get_sigma_effects(
  model = model,
  vc = vc,
  factor_name = "destination",
  factor_levels = levels(df$destination)
)

# =========================================================
# 10. PARAMETER OUTPUT
# =========================================================

parameter_output <- bind_rows(
  intercept_df,
  distance_df,
  cv_df,
  origin_df,
  destination_df,
  model_fit_df
)

# =========================================================
# 11. PREDICTION OUTPUT
# =========================================================

prediction_output <- df %>%
  mutate(
    predicted_flow = predict(model, type = "response"),
    residual = flow - predicted_flow
  )

# =========================================================
# 12. INTEGRATED OUTPUT
# =========================================================

parameter_output2 <- parameter_output %>%
  mutate(
    output_type = "parameter",
    origin = NA,
    destination = NA,
    flow = NA,
    predicted_flow = NA,
    residual = NA,
    distance_value = NA
  ) %>%
  select(
    output_type,
    parameter_type,
    region,
    origin,
    destination,
    coefficient,
    std_error,
    z_value,
    wald_statistic,
    p_value,
    multiplicative_factor,
    flow,
    predicted_flow,
    residual,
    distance_value
  )

prediction_output2 <- prediction_output %>%
  mutate(
    output_type = "prediction",
    parameter_type = NA,
    region = NA,
    coefficient = NA,
    std_error = NA,
    z_value = NA,
    wald_statistic = NA,
    p_value = NA,
    multiplicative_factor = NA,
    distance_value = distance,
    cv_pdrb_value = cv_pdrb 
  ) %>%
  select(
    output_type,
    parameter_type,
    region,
    origin,
    destination,
    coefficient,
    std_error,
    z_value,
    wald_statistic,
    p_value,
    multiplicative_factor,
    flow,
    predicted_flow,
    residual,
    distance_value,
    cv_pdrb_value
  )

final_output <- bind_rows(
  parameter_output2,
  prediction_output2
)

# =========================================================
# 13. SAVE SINGLE CSV
# =========================================================

write_csv(final_output, "entropy_glm_sigma_delta_r2_output_aug.csv")

# =========================================================
# 14. CHECK RESULTS
# =========================================================

cat("\n========================================\n")
cat("MODEL COMPLETED\n")
cat("========================================\n")

cat("\nModel:\n")
cat("log(mu_ij) = intercept + origin_i + destination_j + beta*distance\n")

cat("\nSum origin coefficients:\n")
print(sum(origin_df$coefficient))

cat("\nSum destination coefficients:\n")
print(sum(destination_df$coefficient))

cat("\nModel fit statistics:\n")
print(model_fit_df)

cat("\nOutput saved to:\n")
cat("entropy_glm_sigma_delta_r2_output.csv\n")
