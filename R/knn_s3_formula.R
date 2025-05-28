#' k-NN Regression (S3 + formula interface)
#'
#' @param formula a model formula, y ~ x1 + x2 + ...
#' @param data    a data frame containing the variables
#' @param k       number of neighbors
#' @export
knn_s3 <- function(formula, data, k = 5L) {
  mf <- model.frame(formula, data)
  y  <- model.response(mf)
  X  <- model.matrix(attr(mf, "terms"), mf)
  stopifnot(is.numeric(k), k >= 1, k <= nrow(X))
  structure(
    list(
      call    = match.call(),
      formula = formula,
      train_x = X,
      train_y = y,
      k       = as.integer(k)
    ),
    class = "knn_s3"
  )
}

# print method
#' @export
print.knn_s3 <- function(x, ...) {
  cat("knn_s3 model:\n")
  cat("  Call: "); print(x$call)
  cat("  Training: n =", nrow(x$train_x),
      " p =", ncol(x$train_x),
      " k =", x$k, "\n")
  invisible(x)
}

# summary method: training MSE & R²
#' @export
summary.knn_s3 <- function(object, ...) {
  preds <- fitted(object)
  obs   <- object$train_y
  mse   <- mean((obs - preds)^2)
  r2    <- 1 - sum((obs - preds)^2) / sum((obs - mean(obs))^2)
  cat("Training performance:\n")
  cat("  MSE:", round(mse,4), "   R²:", round(r2,4), "\n")
  invisible(list(call = object$call, k = object$k, MSE = mse, R2 = r2))
}

# fitted values
#' @export
fitted.knn_s3 <- function(object, ...) {
  predict(object, newdata = object$train_x, method = "cpp")
}

#' @export
predict.knn_s3 <- function(object, newdata, method = c("R","cpp"), ...) {
  method <- match.arg(method)
  
  # 1) figure out our test‐matrix Xtest
  if (is.matrix(newdata)) {
    # fitted() will pass us object$train_x directly
    Xtest <- newdata
  } else if (is.data.frame(newdata)) {
    # user passed a data.frame, use the formula to build model.frame & matrix
    mf <- model.frame(delete.response(terms(object$formula)), newdata)
    Xtest <- model.matrix(delete.response(terms(object$formula)), mf)
  } else {
    stop("newdata must be either a matrix or a data.frame")
  }
  
  # 2) dispatch to R or C++ backend
  if (method == "R") {
    n <- nrow(object$train_x)
    m <- nrow(Xtest)
    preds <- numeric(m)
    for (j in seq_len(m)) {
      # squared distances
      d2 <- rowSums((object$train_x -
                       matrix(Xtest[j,], n, ncol(Xtest), byrow = TRUE))^2)
      nn <- order(d2)[1:object$k]
      preds[j] <- mean(object$train_y[nn])
    }
    preds
  } else {
    knn_pred_cpp(object$train_x, object$train_y, Xtest, object$k)
  }
}

# anova method: compare models by training MSE
#' @export
anova.knn_s3 <- function(object, ...) {
  models <- list(object, ...)
  calls  <- sapply(models, function(m) deparse(m$call))
  mses   <- sapply(models, function(m) mean((m$train_y - fitted(m))^2))
  df     <- data.frame(
    Model = calls,
    k     = sapply(models, `[[`, "k"),
    MSE   = mses
  )
  df$DeltaMSE <- c(NA, diff(df$MSE))
  print(df, row.names = FALSE)
  invisible(df)
}
