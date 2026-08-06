/**
 * Duffy vs Adaptive self-quadrature on the twisted cubed sphere.
 *
 * Per (scheme, kernel, tol): self time (SelfInterac, exactly the call SetupSelf makes),
 * total BIOp setup time, and the Green's-identity error so accuracy is matched rather than
 * assumed. The near stage is only reachable through the BIOp, so it is read from the
 * profile tree printed under each block (SetupNear / BuildNearLst).
 *
 *   bench-duffy <order> <patch/face> <tol-list> <twist-list> [threads]
 */

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <sctl.hpp>
#include <sctl/experimental/quad_element.hpp>
#include <sctl/experimental/quad_element.cpp>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace sctl;
using Real = double;

namespace {

constexpr Integer COORD_DIM = 3;

void FacePoint(Real& x, Real& y, Real& z, Integer face, Real a, Real b, Real R) {
  switch (face) {
    case 0: x =  1; y =  a; z =  b; break;
    case 1: x = -1; y = -a; z =  b; break;
    case 2: x =  a; y =  1; z = -b; break;
    case 3: x =  a; y = -1; z =  b; break;
    case 4: x =  a; y =  b; z =  1; break;
    case 5: x = -a; y =  b; z = -1; break;
    default: SCTL_ASSERT(false);
  }
  const Real r = sqrt<Real>(x*x + y*y + z*z);
  x *= R/r; y *= R/r; z *= R/r;
}

QuadElemList<Real> BuildTwistedSphere(Integer ElemOrder, Long PatchPerFace, Real Radius, Real theta_twist) {
  Vector<Real> X;
  static const Long ppv = []() { const char* v = std::getenv("PPF_V"); return v ? atol(v) : (Long)0; }();
  const Long PPU = PatchPerFace, PPV = (ppv > 0 ? ppv : PatchPerFace);
  const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(ElemOrder);
  for (Integer face = 0; face < 6; face++)
    for (Long iu = 0; iu < PPU; iu++)
      for (Long iv = 0; iv < PPV; iv++)
        for (Integer i = 0; i < ElemOrder; i++) {
          const Real a = 2*((iu + nds[i]) / (Real)PPU) - 1;
          for (Integer j = 0; j < ElemOrder; j++) {
            const Real b = 2*((iv + nds[j]) / (Real)PPV) - 1;
            Real x, y, z;
            FacePoint(x, y, z, face, a, b, Radius);
            const Real s = sin<Real>(theta_twist*z), c = cos<Real>(theta_twist*z);
            static const Real sz = []() { const char* v = std::getenv("STRETCH_Z"); return v ? (Real)atof(v) : (Real)1; }();
            X.PushBack(x*c + y*s);
            X.PushBack(-x*s + y*c);
            X.PushBack(z*sz);
          }
        }
  QuadElemList<Real> qel(ElemOrder, X);
  // Fork has 5 schemes (default Adaptive); upstream had only this one. Measure Duffy
  // (upstream self + split-at-foot near) to reproduce the numbers this bench targets.
  qel.SetQuadScheme(QuadElemList<Real>::QuadScheme::Duffy);
  return qel;
}

double Wtime() {
#ifdef _OPENMP
  return omp_get_wtime();
#else
  return (double)clock()/CLOCKS_PER_SEC;
#endif
}

std::vector<double> ParseList(const char* s, const std::vector<double>& dflt) {
  if (!s) return dflt;
  std::vector<double> v;
  const std::string t(s);
  size_t p = 0;
  while (p < t.size()) {
    size_t q = t.find(',', p);
    if (q == std::string::npos) q = t.size();
    v.push_back(atof(t.substr(p, q-p).c_str()));
    p = q + 1;
  }
  return v.empty() ? dflt : v;
}

// Interior on-surface Green's identity: S[t] - D[u] = u with the DL principal-value jump
// removed, driven by a point source OUTSIDE the sphere so the field is a genuine interior
// solution and the density varies across every panel. Laplace uses t = grad(u).n; Stokes uses
// the traction sigma_ij n_j from the FxT stress. The identity couples SL and DL, so accuracy
// is one number per configuration.
// Constant-density identities on a closed surface: D[1] = -1/2 (interior limit) and S[1] is
// the constant 4*pi*a (a=1 here, times the kernel normalisation). Both have exact answers, and
// a constant density removes density interpolation from the picture entirely -- but note they
// only probe the ROW SUMS of the operator, so they are blind to nodal-distribution errors that
// a varying density exposes.
template <class Ker> void ConstDensity(const QuadElemList<Real>& qel, const Real tol, const Comm& comm,
                                       const char* tag, const char* fam, const double twist) {
  static constexpr Integer KDIM0 = Ker::SrcDim();
  Vector<Real> X, Xn; qel.GetNodeCoord(&X,&Xn,nullptr);
  const Long N = X.Dim()/COORD_DIM;
  Vector<Real> F(N*KDIM0); F.SetZero();
  for (Long i = 0; i < N; i++) F[i*KDIM0] = 1;   // first component unit, rest zero
  BoundaryIntegralOp<Real,Ker> B(Ker(), false, comm);
  B.SetAccuracy(tol); B.AddElemList(qel); B.Setup();
  Vector<Real> U; B.ComputePotential(U, F);
  // component 0 of the response to a unit density in component 0
  double mn=1e300, mx=-1e300, sum=0;
  for (Long i = 0; i < N; i++) { const double v=(double)U[i*KDIM0]; mn=std::min(mn,v); mx=std::max(mx,v); sum+=v; }
  const double mean = sum/(double)N;
  const double spread = std::max(std::fabs(mx-mean), std::fabs(mean-mn));
  double dl_err = 0;
  for (Long i = 0; i < N; i++) dl_err = std::max(dl_err, std::fabs((double)U[i*KDIM0] + 0.5));
  if (const char* cd = std::getenv("CD_DUMP")) { // per-node |D[1]+1/2| as VTU + text
    if (std::string(tag) == "DL[1]") {
      Vector<Real> E(N);
      for (Long i = 0; i < N; i++) E[i] = fabs<Real>(U[i*KDIM0] + (Real)0.5);
      qel.WriteVTK(std::string(cd)+"_"+std::string(fam), E);
      FILE* fp = fopen((std::string(cd)+"_"+std::string(fam)+".txt").c_str(),"w");
      if (fp) { for (Long i = 0; i < N; i++)
          fprintf(fp,"%.9f %.9f %.9f %.6e\n",(double)X[i*COORD_DIM+0],(double)X[i*COORD_DIM+1],
                  (double)X[i*COORD_DIM+2],(double)E[i]);
        fclose(fp); }
    }
  }
  const bool is_dl = (std::string(tag) == "DL[1]");
  std::printf("#CD %-7s %-8s tol=%-8.0e mean=%12.8f  spread=%9.2e  rel=%9.2e%s\n",
              fam, tag, (double)tol, mean, spread, spread/std::max(1e-300,std::fabs(mean)),
              is_dl ? ("   |D[1]+1/2|max=" + [&]{ char b[32]; snprintf(b,sizeof b,"%9.2e",dl_err); return std::string(b); }()).c_str() : "");
  (void)twist;
  std::fflush(stdout);
}

template <class KerSL, class KerDL, class KerGrad>
double GreensErr(const QuadElemList<Real>& qel, const Real tol, const Comm& comm,
                 double* t_setup_sl, double* t_setup_dl) {
  static constexpr Integer KDIM0 = KerSL::SrcDim();
  static constexpr Integer GDIM  = KerGrad::TrgDim();   // 3 (Laplace grad) or 9 (Stokes stress)
  KerSL ker_sl; KerDL ker_dl; KerGrad ker_grad;

  Vector<Real> X, Xn;
  qel.GetNodeCoord(&X, &Xn, nullptr);
  const Long N = X.Dim()/COORD_DIM;
  const Vector<Real> X0{(Real)1.3, (Real)1.2, (Real)0.2};
  Vector<Real> Xn0(COORD_DIM); Xn0[0]=0; Xn0[1]=0; Xn0[2]=1;
  Vector<Real> F0(KDIM0), dU, Fd, Uref, Fs;
  for (Integer i = 0; i < KDIM0; i++) F0[i] = (Real)1/(Real)(i+1);
  ker_sl.Eval(Fd, X, X0, Xn0, F0);
  ker_grad.Eval(dU, X, X0, Xn0, F0);
  ker_sl.Eval(Uref, X, X0, Xn0, F0);

  Fs.ReInit(N*KDIM0);
  for (Long i = 0; i < N; i++) {
    if (GDIM == COORD_DIM) {                       // Laplace: du/dn
      Real dn = 0;
      for (Integer k = 0; k < COORD_DIM; k++) dn += dU[i*GDIM+k]*Xn[i*COORD_DIM+k];
      Fs[i] = dn;
    } else {                                       // Stokes: traction sigma_ij n_j
      for (Integer k = 0; k < COORD_DIM; k++) {
        Real t = 0;
        for (Integer j = 0; j < COORD_DIM; j++) t += dU[i*GDIM + k*COORD_DIM + j]*Xn[i*COORD_DIM+j];
        Fs[i*KDIM0+k] = t;
      }
    }
  }

  BoundaryIntegralOp<Real,KerSL> BSL(ker_sl, false, comm);
  BoundaryIntegralOp<Real,KerDL> BDL(ker_dl, false, comm);
  BSL.AddElemList(qel); BDL.AddElemList(qel);
  BSL.SetAccuracy(tol); BDL.SetAccuracy(tol);
  double t0 = Wtime(); BSL.Setup(); if (t_setup_sl) *t_setup_sl = Wtime()-t0;
  t0 = Wtime();         BDL.Setup(); if (t_setup_dl) *t_setup_dl = Wtime()-t0;

  Vector<Real> Us, Ud;
  BSL.ComputePotential(Us, Fs);
  BDL.ComputePotential(Ud, Fd);
  Ud -= (Real)0.5*Fd;
  double e = 0, m = 0;
  for (Long i = 0; i < Uref.Dim(); i++) {
    e = std::max(e, (double)fabs<Real>((Us[i]-Ud[i]) - Uref[i]));
    m = std::max(m, (double)fabs<Real>(Uref[i]));
  }
  if (const char* fn = std::getenv("ERR_DUMP")) { // per-node error field for visualisation
    FILE* fp = fopen(fn, "w");
    if (fp) {
      for (Long i = 0; i < N; i++) {
        double ei = 0;
        for (Integer k = 0; k < KDIM0; k++) {
          const Long t = i*KDIM0 + k;
          ei = std::max(ei, (double)fabs<Real>((Us[t]-Ud[t]) - Uref[t]));
        }
        fprintf(fp, "%.9f %.9f %.9f %.6e\n", (double)X[i*COORD_DIM+0], (double)X[i*COORD_DIM+1],
                (double)X[i*COORD_DIM+2], ei/(m>0?m:1.0));
      }
      fclose(fp);
    }
    // same field as a VTU so it can be opened in ParaView
    Vector<Real> Efld(N);
    for (Long i = 0; i < N; i++) {
      Real ei = 0;
      for (Integer k = 0; k < KDIM0; k++) {
        const Long t = i*KDIM0 + k;
        ei = std::max<Real>(ei, fabs<Real>((Us[t]-Ud[t]) - Uref[t]));
      }
      Efld[i] = ei/(Real)(m > 0 ? m : 1.0);
    }
    std::string vn(fn);
    const size_t dot = vn.rfind(".txt");
    if (dot != std::string::npos) vn = vn.substr(0, dot);
    qel.WriteVTK(vn, Efld);
  }
  return (m > 0 ? e/m : 0.0);
}

template <class Kernel>
double TimeSelf(QuadElemList<Real>& qel, const Real tol) {
  const Kernel ker;
  Vector<Matrix<Real>> M_lst(qel.Size());
  QuadElemList<Real>::template SelfInterac<Kernel>(M_lst, ker, tol, false, &qel); // warm
  double t = 1e30;
  for (int r = 0; r < 3; r++) {
    const double t0 = Wtime();
    QuadElemList<Real>::template SelfInterac<Kernel>(M_lst, ker, tol, false, &qel);
    t = std::min(t, Wtime()-t0);
  }
  return t;
}

template <class KerSL, class KerDL, class KerGrad>
void RunCfg(const char* scheme, const char* fam, QuadElemList<Real>& qel, const Real tol,
            const double twist, const Comm& comm, const Integer nthr) {
  const Long nnode = qel.Size()*qel.Order()*qel.Order();
  const double ts_sl = TimeSelf<KerSL>(qel, tol);
  const double ts_dl = TimeSelf<KerDL>(qel, tol);
  { BoundaryIntegralOp<Real,KerSL> A(KerSL(), false, comm);
    BoundaryIntegralOp<Real,KerDL> B(KerDL(), false, comm);
    A.SetAccuracy(tol); A.AddElemList(qel); A.Setup();
    B.SetAccuracy(tol); B.AddElemList(qel); B.Setup(); }
  double tu_sl = 0, tu_dl = 0;
  const double err = GreensErr<KerSL,KerDL,KerGrad>(qel, tol, comm, &tu_sl, &tu_dl);
  const double n = (double)nnode, T = (double)nthr;
  std::printf("#ROW %-9s %-7s tol=%-8.0e twist=%-8.4f err=%10.3e "
              "self_sl=%8.4f self_dl=%8.4f setup_sl=%8.4f setup_dl=%8.4f "
              "pps_self_sl=%9.1f pps_self_dl=%9.1f pps_setup_sl=%9.1f pps_setup_dl=%9.1f\n",
              scheme, fam, (double)tol, twist, err, ts_sl, ts_dl, tu_sl, tu_dl,
              n/ts_sl/T, n/ts_dl/T, n/tu_sl/T, n/tu_dl/T);
  std::fflush(stdout);
}

} // namespace

int main(int argc, char** argv) {
  Comm::MPI_Init(&argc, &argv);
  {
    const Comm comm = Comm::World();
    const Integer order     = (argc > 1 ? (Integer)atol(argv[1]) : 12);
    const Long ppf          = (argc > 2 ? atol(argv[2]) : 8);
    const std::vector<double> tols   = ParseList(argc > 3 ? argv[3] : nullptr, {1e-6, 1e-8, 1e-10, 1e-12});
    const std::vector<double> twists = ParseList(argc > 4 ? argv[4] : nullptr, {0.0, 0.5235988, 1.0471976, 1.5707963});
    const Integer nthr      = (argc > 5 ? (Integer)atol(argv[5]) : SCTL_GET_MAX_THREADS());
#ifdef _OPENMP
    omp_set_num_threads(nthr);
#endif
    std::printf("==== Duffy self / split near: order=%d ppf=%ld threads=%d ====\n", (int)order, ppf, (int)nthr);

    for (const double tw : twists) {
      QuadElemList<Real> q = BuildTwistedSphere(order, ppf, 1.0, (Real)tw);
      for (const double tol : tols) {
        if (std::getenv("CONST_DENSITY")) {
          ConstDensity<Laplace3D_FxU>(q,(Real)tol,comm,"SL[1]","laplace",tw);
          ConstDensity<Laplace3D_DxU>(q,(Real)tol,comm,"DL[1]","laplace",tw);
          ConstDensity<Stokes3D_FxU>(q,(Real)tol,comm,"SL[1]","stokes",tw);
          ConstDensity<Stokes3D_DxU>(q,(Real)tol,comm,"DL[1]","stokes",tw);
          continue;
        }
        RunCfg<Laplace3D_FxU,Laplace3D_DxU,Laplace3D_FxdU>("duffy","laplace",q,(Real)tol,tw,comm,nthr);
        RunCfg<Stokes3D_FxU, Stokes3D_DxU, Stokes3D_FxT >("duffy","stokes", q,(Real)tol,tw,comm,nthr);
      }
    }
  }
  Comm::MPI_Finalize();
  return 0;
}
