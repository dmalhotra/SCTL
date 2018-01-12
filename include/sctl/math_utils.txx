#include <omp.h>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <vector>

namespace SCTL_NAMESPACE {

template <> inline unsigned int pow(const unsigned int b, const unsigned int e) {
  unsigned int r = 1;
  for (unsigned int i = 0; i < e; i++) r *= b;
  return r;
}
}

#ifdef SCTL_QUAD_T

namespace SCTL_NAMESPACE {

template <> inline QuadReal const_pi<QuadReal>() {
  static QuadReal pi = atoquad("3.1415926535897932384626433832795028841");
  return pi;
}

template <> inline QuadReal const_e<QuadReal>() {
  static QuadReal e = atoquad("2.7182818284590452353602874713526624977");
  return e;
}

template <> inline QuadReal fabs(const QuadReal f) {
  if (f >= 0.0)
    return f;
  else
    return -f;
}

template <> inline QuadReal sqrt(const QuadReal a) {
  QuadReal b = ::sqrt((double)a);
  b = (b + a / b) * 0.5;
  b = (b + a / b) * 0.5;
  return b;
}

template <> inline QuadReal sin(const QuadReal a) {
  const int N = 200;
  static std::vector<QuadReal> theta;
  static std::vector<QuadReal> sinval;
  static std::vector<QuadReal> cosval;
  if (theta.size() == 0) {
#pragma omp critical(QUAD_SIN)
    if (theta.size() == 0) {
      theta.resize(N);
      sinval.resize(N);
      cosval.resize(N);

      QuadReal t = 1.0;
      for (int i = 0; i < N; i++) {
        theta[i] = t;
        t = t * 0.5;
      }

      sinval[N - 1] = theta[N - 1];
      cosval[N - 1] = 1.0 - sinval[N - 1] * sinval[N - 1] / 2;
      for (int i = N - 2; i >= 0; i--) {
        sinval[i] = 2.0 * sinval[i + 1] * cosval[i + 1];
        cosval[i] = sqrt<QuadReal>(1.0 - sinval[i] * sinval[i]);
      }
    }
  }

  QuadReal t = (a < 0.0 ? -a : a);
  QuadReal sval = 0.0;
  QuadReal cval = 1.0;
  for (int i = 0; i < N; i++) {
    while (theta[i] <= t) {
      QuadReal sval_ = sval * cosval[i] + cval * sinval[i];
      QuadReal cval_ = cval * cosval[i] - sval * sinval[i];
      sval = sval_;
      cval = cval_;
      t = t - theta[i];
    }
  }
  return (a < 0.0 ? -sval : sval);
}

template <> inline QuadReal cos(const QuadReal a) {
  const int N = 200;
  static std::vector<QuadReal> theta;
  static std::vector<QuadReal> sinval;
  static std::vector<QuadReal> cosval;
  if (theta.size() == 0) {
#pragma omp critical(QUAD_COS)
    if (theta.size() == 0) {
      theta.resize(N);
      sinval.resize(N);
      cosval.resize(N);

      QuadReal t = 1.0;
      for (int i = 0; i < N; i++) {
        theta[i] = t;
        t = t * 0.5;
      }

      sinval[N - 1] = theta[N - 1];
      cosval[N - 1] = 1.0 - sinval[N - 1] * sinval[N - 1] / 2;
      for (int i = N - 2; i >= 0; i--) {
        sinval[i] = 2.0 * sinval[i + 1] * cosval[i + 1];
        cosval[i] = sqrt<QuadReal>(1.0 - sinval[i] * sinval[i]);
      }
    }
  }

  QuadReal t = (a < 0.0 ? -a : a);
  QuadReal sval = 0.0;
  QuadReal cval = 1.0;
  for (int i = 0; i < N; i++) {
    while (theta[i] <= t) {
      QuadReal sval_ = sval * cosval[i] + cval * sinval[i];
      QuadReal cval_ = cval * cosval[i] - sval * sinval[i];
      sval = sval_;
      cval = cval_;
      t = t - theta[i];
    }
  }
  return cval;
}

template <> inline QuadReal exp(const QuadReal a) {
  const int N = 200;
  static std::vector<QuadReal> theta0;
  static std::vector<QuadReal> theta1;
  static std::vector<QuadReal> expval0;
  static std::vector<QuadReal> expval1;
  if (theta0.size() == 0) {
#pragma omp critical(QUAD_EXP)
    if (theta0.size() == 0) {
      theta0.resize(N);
      theta1.resize(N);
      expval0.resize(N);
      expval1.resize(N);

      theta0[0] = 1.0;
      theta1[0] = 1.0;
      expval0[0] = const_e<QuadReal>();
      expval1[0] = const_e<QuadReal>();
      for (int i = 1; i < N; i++) {
        theta0[i] = theta0[i - 1] * 0.5;
        theta1[i] = theta1[i - 1] * 2.0;
        expval0[i] = sqrt<QuadReal>(expval0[i - 1]);
        expval1[i] = expval1[i - 1] * expval1[i - 1];
      }
    }
  }

  QuadReal t = (a < 0.0 ? -a : a);
  QuadReal eval = 1.0;
  for (int i = N - 1; i > 0; i--) {
    while (theta1[i] <= t) {
      eval = eval * expval1[i];
      t = t - theta1[i];
    }
  }
  for (int i = 0; i < N; i++) {
    while (theta0[i] <= t) {
      eval = eval * expval0[i];
      t = t - theta0[i];
    }
  }
  eval = eval * (1.0 + t);
  return (a < 0.0 ? 1.0 / eval : eval);
}

template <> inline QuadReal log(const QuadReal a) {
  QuadReal y0 = ::log((double)a);
  y0 = y0 + (a / exp<QuadReal>(y0) - 1.0);
  y0 = y0 + (a / exp<QuadReal>(y0) - 1.0);
  return y0;
}

template <> inline QuadReal pow(const QuadReal b, const QuadReal e) {
  if (b == 0) return 1;
  return exp<QuadReal>(log<QuadReal>(b) * e);
}

inline QuadReal atoquad(const char* str) {
  int i = 0;
  QuadReal sign = 1.0;
  for (; str[i] != '\0'; i++) {
    char c = str[i];
    if (c == '-') sign = -sign;
    if (c >= '0' && c <= '9') break;
  }

  QuadReal val = 0.0;
  for (; str[i] != '\0'; i++) {
    char c = str[i];
    if (c >= '0' && c <= '9')
      val = val * 10 + (c - '0');
    else
      break;
  }

  if (str[i] == '.') {
    i++;
    QuadReal exp = 1.0;
    exp /= 10;
    for (; str[i] != '\0'; i++) {
      char c = str[i];
      if (c >= '0' && c <= '9')
        val = val + (c - '0') * exp;
      else
        break;
      exp /= 10;
    }
  }

  return sign * val;
}

inline std::ostream& operator<<(std::ostream& output, const QuadReal q_) {
  // int width=output.width();
  output << std::setw(1);

  QuadReal q = q_;
  if (q < 0.0) {
    output << "-";
    q = -q;
  } else if (q > 0) {
    output << " ";
  } else {
    output << " ";
    output << "0.0";
    return output;
  }

  int exp = 0;
  static const QuadReal ONETENTH = (QuadReal)1 / 10;
  while (q < 1.0 && abs(exp) < 10000) {
    q = q * 10;
    exp--;
  }
  while (q >= 10 && abs(exp) < 10000) {
    q = q * ONETENTH;
    exp++;
  }

  for (int i = 0; i < 34; i++) {
    output << (int)q;
    if (i == 0) output << ".";
    q = (q - int(q)) * 10;
    if (q == 0 && i > 0) break;
  }

  if (exp > 0) {
    std::cout << "e+" << exp;
  } else if (exp < 0) {
    std::cout << "e" << exp;
  }

  return output;
}

}  // end namespace

#endif  // SCTL_QUAD_T
