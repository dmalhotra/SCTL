#include SCTL_INCLUDE(legendre_rule.hpp)

namespace SCTL_NAMESPACE {

//    Vector<Real> qx1, qw1;
//    //cgqf(p0+1, 1, 0.0, 0.0, -1.0, 1.0, &qx1[0], &qw1[0]);
//    ChebBasis<Real>::quad_rule(p0+1, qx1, qw1);
//    sctl::ASSERT(typeid(Real) == typeid(double) || typeid(Real) == typeid(float)); // TODO: these are Legendre nodes only for float and double
//    for (auto x : qx1) x = 2 * x - 1;
//    for (auto w : qw1) w = 2 * w;


template <class Real> void SphericalHarmonics<Real>::SHC2Grid(const Vector<Real>& S, Long p0, Long p1, Vector<Real>& X, Vector<Real>* X_theta, Vector<Real>* X_phi){
  Matrix<Real>& Mf =SphericalHarmonics<Real>::MatFourier(p0,p1);
  Matrix<Real>& Mdf=SphericalHarmonics<Real>::MatFourierGrad(p0,p1);
  std::vector<Matrix<Real>>& Ml =SphericalHarmonics<Real>::MatLegendre(p0,p1);
  std::vector<Matrix<Real>>& Mdl=SphericalHarmonics<Real>::MatLegendreGrad(p0,p1);
  assert(p0==(Long)Ml.size()-1);
  assert(p0==Mf.Dim(0)/2);
  assert(p1==Mf.Dim(1)/2);

  Long N=S.Dim()/(p0*(p0+2));
  assert(N*p0*(p0+2)==S.Dim());

  if(X.Dim()!=N*2*p1*(p1+1)) X.ReInit(N*2*p1*(p1+1));
  if(X_phi   && X_phi  ->Dim()!=N*2*p1*(p1+1)) X_phi  ->ReInit(N*2*p1*(p1+1));
  if(X_theta && X_theta->Dim()!=N*2*p1*(p1+1)) X_theta->ReInit(N*2*p1*(p1+1));

  static Vector<Real> B0, B1;
  B0.ReInit(N*  p0*(p0+2));
  B1.ReInit(N*2*p0*(p1+1));

  #pragma omp parallel
  { // B0 <-- Rearrange(S)
    Integer tid=omp_get_thread_num();
    Integer omp_p=omp_get_num_threads();

    Long a=(tid+0)*N/omp_p;
    Long b=(tid+1)*N/omp_p;
    for(Long i=a;i<b;i++){
      Long offset=0;
      for(Long j=0;j<2*p0;j++){
        Long len=p0+1-(j+1)/2;
        Real* B_=&B0[i*len+N*offset];
        const Real* S_=&S[i*p0*(p0+2)+offset];
        for(Long k=0;k<len;k++) B_[k]=S_[k];
        offset+=len;
      }
    }
  }

  #pragma omp parallel
  { // Evaluate Legendre polynomial
    Integer tid=omp_get_thread_num();
    Integer omp_p=omp_get_num_threads();

    Long offset0=0;
    Long offset1=0;
    for(Long i=0;i<p0+1;i++){
      Long N0=2*N;
      if(i==0 || i==p0) N0=N;
      Matrix<Real> Min (N0, p0+1-i, B0.begin()+offset0, false);
      Matrix<Real> Mout(N0, p1+1  , B1.begin()+offset1, false);
      { // Mout = Min * Ml[i]  // split between threads
        Long a=(tid+0)*N0/omp_p;
        Long b=(tid+1)*N0/omp_p;
        if(a<b){
          Matrix<Real> Min_ (b-a, Min .Dim(1), Min [a], false);
          Matrix<Real> Mout_(b-a, Mout.Dim(1), Mout[a], false);
          Matrix<Real>::GEMM(Mout_,Min_,Ml[i]);
        }
      }
      offset0+=Min .Dim(0)*Min .Dim(1);
      offset1+=Mout.Dim(0)*Mout.Dim(1);
    }
  }

  #pragma omp parallel
  { // Transpose and evaluate Fourier
    Integer tid=omp_get_thread_num();
    Integer omp_p=omp_get_num_threads();

    Long a=(tid+0)*N*(p1+1)/omp_p;
    Long b=(tid+1)*N*(p1+1)/omp_p;

    const Long block_size=16;
    Matrix<Real> B2(block_size,2*p0);
    for(Long i0=a;i0<b;i0+=block_size){
      Long i1=std::min(b,i0+block_size);
      for(Long i=i0;i<i1;i++){
        for(Long j=0;j<2*p0;j++){
          B2[i-i0][j]=B1[j*N*(p1+1)+i];
        }
      }

      Matrix<Real> Min (i1-i0,2*p0, B2.begin()        , false);
      Matrix<Real> Mout(i1-i0,2*p1, X .begin()+i0*2*p1, false);
      Matrix<Real>::GEMM(Mout, Min, Mf);

      if(X_theta){ // Evaluate Fourier gradient
        Matrix<Real> Mout(i1-i0,2*p1, X_theta->begin()+i0*2*p1, false);
        Matrix<Real>::GEMM(Mout, Min, Mdf);
      }
    }
  }

  if(X_phi){
    #pragma omp parallel
    { // Evaluate Legendre gradient
      Integer tid=omp_get_thread_num();
      Integer omp_p=omp_get_num_threads();

      Long offset0=0;
      Long offset1=0;
      for(Long i=0;i<p0+1;i++){
        Long N0=2*N;
        if(i==0 || i==p0) N0=N;
        Matrix<Real> Min (N0, p0+1-i, B0.begin()+offset0, false);
        Matrix<Real> Mout(N0, p1+1  , B1.begin()+offset1, false);
        { // Mout = Min * Mdl[i]  // split between threads
          Long a=(tid+0)*N0/omp_p;
          Long b=(tid+1)*N0/omp_p;
          if(a<b){
            Matrix<Real> Min_ (b-a, Min .Dim(1), Min [a], false);
            Matrix<Real> Mout_(b-a, Mout.Dim(1), Mout[a], false);
            Matrix<Real>::GEMM(Mout_,Min_,Mdl[i]);
          }
        }
        offset0+=Min .Dim(0)*Min .Dim(1);
        offset1+=Mout.Dim(0)*Mout.Dim(1);
      }
    }

    #pragma omp parallel
    { // Transpose and evaluate Fourier
      Integer tid=omp_get_thread_num();
      Integer omp_p=omp_get_num_threads();

      Long a=(tid+0)*N*(p1+1)/omp_p;
      Long b=(tid+1)*N*(p1+1)/omp_p;

      const Long block_size=16;
      Matrix<Real> B2(block_size,2*p0);
      for(Long i0=a;i0<b;i0+=block_size){
        Long i1=std::min(b,i0+block_size);
        for(Long i=i0;i<i1;i++){
          for(Long j=0;j<2*p0;j++){
            B2[i-i0][j]=B1[j*N*(p1+1)+i];
          }
        }

        Matrix<Real> Min (i1-i0,2*p0, B2.begin()            , false);
        Matrix<Real> Mout(i1-i0,2*p1, X_phi->begin()+i0*2*p1, false);
        Matrix<Real>::GEMM(Mout, Min, Mf);
      }
    }
  }
}

template <class Real> void SphericalHarmonics<Real>::Grid2SHC(const Vector<Real>& X, Long p0, Long p1, Vector<Real>& S){
  Matrix<Real> Mf =SphericalHarmonics<Real>::MatFourierInv(p0,p1);
  std::vector<Matrix<Real>> Ml =SphericalHarmonics<Real>::MatLegendreInv(p0,p1);
  assert(p1==(Long)Ml.size()-1);
  assert(p0==Mf.Dim(0)/2);
  assert(p1==Mf.Dim(1)/2);

  Long N=X.Dim()/(2*p0*(p0+1));
  assert(N*2*p0*(p0+1)==X.Dim());
  if(S.Dim()!=N*(p1*(p1+2))) S.ReInit(N*(p1*(p1+2)));

  static Vector<Real> B0, B1;
  B0.ReInit(N*  p1*(p1+2));
  B1.ReInit(N*2*p1*(p0+1));

  #pragma omp parallel
  { // Evaluate Fourier and transpose
    Integer tid=omp_get_thread_num();
    Integer omp_p=omp_get_num_threads();

    Long a=(tid+0)*N*(p0+1)/omp_p;
    Long b=(tid+1)*N*(p0+1)/omp_p;

    const Long block_size=16;
    Matrix<Real> B2(block_size,2*p1);
    for(Long i0=a;i0<b;i0+=block_size){
      Long i1=std::min(b,i0+block_size);
      Matrix<Real> Min (i1-i0,2*p0, (Iterator<Real>)X.begin()+i0*2*p0, false);
      Matrix<Real> Mout(i1-i0,2*p1, B2.begin()                       , false);
      Matrix<Real>::GEMM(Mout, Min, Mf);

      for(Long i=i0;i<i1;i++){
        for(Long j=0;j<2*p1;j++){
          B1[j*N*(p0+1)+i]=B2[i-i0][j];
        }
      }
    }
  }

  #pragma omp parallel
  { // Evaluate Legendre polynomial
    Integer tid=omp_get_thread_num();
    Integer omp_p=omp_get_num_threads();

    Long offset0=0;
    Long offset1=0;
    for(Long i=0;i<p1+1;i++){
      Long N0=2*N;
      if(i==0 || i==p1) N0=N;
      Matrix<Real> Min (N0, p0+1  , B1.begin()+offset0, false);
      Matrix<Real> Mout(N0, p1+1-i, B0.begin()+offset1, false);
      { // Mout = Min * Ml[i]  // split between threads
        Long a=(tid+0)*N0/omp_p;
        Long b=(tid+1)*N0/omp_p;
        if(a<b){
          Matrix<Real> Min_ (b-a, Min .Dim(1), Min [a], false);
          Matrix<Real> Mout_(b-a, Mout.Dim(1), Mout[a], false);
          Matrix<Real>::GEMM(Mout_,Min_,Ml[i]);
        }
      }
      offset0+=Min .Dim(0)*Min .Dim(1);
      offset1+=Mout.Dim(0)*Mout.Dim(1);
    }
  }

  #pragma omp parallel
  { // S <-- Rearrange(B0)
    Integer tid=omp_get_thread_num();
    Integer omp_p=omp_get_num_threads();

    Long a=(tid+0)*N/omp_p;
    Long b=(tid+1)*N/omp_p;
    for(Long i=a;i<b;i++){
      Long offset=0;
      for(Long j=0;j<2*p1;j++){
        Long len=p1+1-(j+1)/2;
        Real* B_=&B0[i*len+N*offset];
        Real* S_=&S[i*p1*(p1+2)+offset];
        for(Long k=0;k<len;k++) S_[k]=B_[k];
        offset+=len;
      }
    }
  }
}

template <class Real> void SphericalHarmonics<Real>::SHC2GridTranspose(const Vector<Real>& X, Long p0, Long p1, Vector<Real>& S){
  Matrix<Real> Mf =SphericalHarmonics<Real>::MatFourier(p1,p0).Transpose();
  std::vector<Matrix<Real>> Ml =SphericalHarmonics<Real>::MatLegendre(p1,p0);
  for(Long i=0;i<(Long)Ml.size();i++) Ml[i]=Ml[i].Transpose();
  assert(p1==(Long)Ml.size()-1);
  assert(p0==Mf.Dim(0)/2);
  assert(p1==Mf.Dim(1)/2);

  Long N=X.Dim()/(2*p0*(p0+1));
  assert(N*2*p0*(p0+1)==X.Dim());
  if(S.Dim()!=N*(p1*(p1+2))) S.ReInit(N*(p1*(p1+2)));

  static Vector<Real> B0, B1;
  B0.ReInit(N*  p1*(p1+2));
  B1.ReInit(N*2*p1*(p0+1));

  #pragma omp parallel
  { // Evaluate Fourier and transpose
    Integer tid=omp_get_thread_num();
    Integer omp_p=omp_get_num_threads();

    Long a=(tid+0)*N*(p0+1)/omp_p;
    Long b=(tid+1)*N*(p0+1)/omp_p;

    const Long block_size=16;
    Matrix<Real> B2(block_size,2*p1);
    for(Long i0=a;i0<b;i0+=block_size){
      Long i1=std::min(b,i0+block_size);
      Matrix<Real> Min (i1-i0,2*p0, (Iterator<Real>)X.begin()+i0*2*p0, false);
      Matrix<Real> Mout(i1-i0,2*p1, B2.begin()                       , false);
      Matrix<Real>::GEMM(Mout, Min, Mf);

      for(Long i=i0;i<i1;i++){
        for(Long j=0;j<2*p1;j++){
          B1[j*N*(p0+1)+i]=B2[i-i0][j];
        }
      }
    }
  }

  #pragma omp parallel
  { // Evaluate Legendre polynomial
    Integer tid=omp_get_thread_num();
    Integer omp_p=omp_get_num_threads();

    Long offset0=0;
    Long offset1=0;
    for(Long i=0;i<p1+1;i++){
      Long N0=2*N;
      if(i==0 || i==p1) N0=N;
      Matrix<Real> Min (N0, p0+1  , B1.begin()+offset0, false);
      Matrix<Real> Mout(N0, p1+1-i, B0.begin()+offset1, false);
      { // Mout = Min * Ml[i]  // split between threads
        Long a=(tid+0)*N0/omp_p;
        Long b=(tid+1)*N0/omp_p;
        if(a<b){
          Matrix<Real> Min_ (b-a, Min .Dim(1), Min [a], false);
          Matrix<Real> Mout_(b-a, Mout.Dim(1), Mout[a], false);
          Matrix<Real>::GEMM(Mout_,Min_,Ml[i]);
        }
      }
      offset0+=Min .Dim(0)*Min .Dim(1);
      offset1+=Mout.Dim(0)*Mout.Dim(1);
    }
  }

  #pragma omp parallel
  { // S <-- Rearrange(B0)
    Integer tid=omp_get_thread_num();
    Integer omp_p=omp_get_num_threads();

    Long a=(tid+0)*N/omp_p;
    Long b=(tid+1)*N/omp_p;
    for(Long i=a;i<b;i++){
      Long offset=0;
      for(Long j=0;j<2*p1;j++){
        Long len=p1+1-(j+1)/2;
        Real* B_=&B0[i*len+N*offset];
        Real* S_=&S[i*p1*(p1+2)+offset];
        for(Long k=0;k<len;k++) S_[k]=B_[k];
        offset+=len;
      }
    }
  }
}

template <class Real> void SphericalHarmonics<Real>::SHC2Pole(const Vector<Real>& S, Long p0, Vector<Real>& P){
  Vector<Real> QP[2];
  { // Set QP
    Real x[2]={-1,1};
    Vector<Real> alp((p0+1)*(p0+2)/2);
    const Real SQRT2PI=sqrt(2*M_PI);
    for(Long i=0;i<2;i++){
      LegPoly(&alp[0], &x[i], 1, p0);
      QP[i].ReInit(p0+1, alp.begin());
      for(Long j=0;j<p0+1;j++) QP[i][j]*=SQRT2PI;
    }
  }

  Long N=S.Dim()/(p0*(p0+2));
  assert(N*p0*(p0+2)==S.Dim());
  if(P.Dim()!=N*2) P.ReInit(N*2);

  #pragma omp parallel
  { // Compute pole
    Integer tid=omp_get_thread_num();
    Integer omp_p=omp_get_num_threads();

    Long a=(tid+0)*N/omp_p;
    Long b=(tid+1)*N/omp_p;

    for(Long i=a;i<b;i++){
      Real P_[2]={0,0};
      for(Long j=0;j<p0+1;j++){
        P_[0]+=S[i*p0*(p0+2)+j]*QP[0][j];
        P_[1]+=S[i*p0*(p0+2)+j]*QP[1][j];
      }
      P[2*i+0]=P_[0];
      P[2*i+1]=P_[1];
    }
  }
}

template <class Real> void SphericalHarmonics<Real>::RotateAll(const Vector<Real>& S, Long p0, Long dof, Vector<Real>& S_){
  std::vector<Matrix<Real>>& Mr=MatRotate(p0);
  std::vector<std::vector<Long>> coeff_perm(p0+1);
  { // Set coeff_perm
    for(Long n=0;n<=p0;n++) coeff_perm[n].resize(std::min(2*n+1,2*p0));
    Long itr=0;
    for(Long i=0;i<2*p0;i++){
      Long m=(i+1)/2;
      for(Long n=m;n<=p0;n++){
        coeff_perm[n][i]=itr;
        itr++;
      }
    }
  }
  Long Ncoef=p0*(p0+2);

  Long N=S.Dim()/Ncoef/dof;
  assert(N*Ncoef*dof==S.Dim());
  if(S_.Dim()!=N*dof*Ncoef*p0*(p0+1)) S_.ReInit(N*dof*Ncoef*p0*(p0+1));
  Matrix<Real> S0(N*dof          ,Ncoef, (Iterator<Real>)S.begin(), false);
  Matrix<Real> S1(N*dof*p0*(p0+1),Ncoef, S_.begin()               , false);

  #pragma omp parallel
  { // Construct all p0*(p0+1) rotations
    Integer tid=omp_get_thread_num();
    Integer omp_p=omp_get_num_threads();
    Matrix<Real> B0(dof*p0,Ncoef); // memory buffer

    std::vector<Matrix<Real>> Bi(p0+1), Bo(p0+1); // memory buffers
    for(Long i=0;i<=p0;i++){ // initialize Bi, Bo
      Bi[i].ReInit(dof*p0,coeff_perm[i].size());
      Bo[i].ReInit(dof*p0,coeff_perm[i].size());
    }

    Long a=(tid+0)*N/omp_p;
    Long b=(tid+1)*N/omp_p;
    for(Long i=a;i<b;i++){
      for(Long d=0;d<dof;d++){
        for(Long j=0;j<p0;j++){
          Long offset=0;
          for(Long k=0;k<p0+1;k++){
            Real r[2]={cos(k*j*M_PI/p0),-sin(k*j*M_PI/p0)}; // exp(i*k*theta)
            Long len=p0+1-k;
            if(k!=0 && k!=p0){
              for(Long l=0;l<len;l++){
                Real x[2];
                x[0]=S0[i*dof+d][offset+len*0+l];
                x[1]=S0[i*dof+d][offset+len*1+l];
                B0[j*dof+d][offset+len*0+l]=x[0]*r[0]-x[1]*r[1];
                B0[j*dof+d][offset+len*1+l]=x[0]*r[1]+x[1]*r[0];
              }
              offset+=2*len;
            }else{
              for(Long l=0;l<len;l++){
                B0[j*dof+d][offset+l]=S0[i*dof+d][offset+l];
              }
              offset+=len;
            }
          }
          assert(offset==Ncoef);
        }
      }
      { // Fast rotation
        for(Long k=0;k<dof*p0;k++){ // forward permutation
          for(Long l=0;l<=p0;l++){
            for(Long j=0;j<(Long)coeff_perm[l].size();j++){
              Bi[l][k][j]=B0[k][coeff_perm[l][j]];
            }
          }
        }
        for(Long t=0;t<=p0;t++){
          for(Long l=0;l<=p0;l++){ // mat-vec
            Matrix<Real>::GEMM(Bo[l],Bi[l],Mr[t*(p0+1)+l]);
          }
          Matrix<Real> Mout(dof*p0,Ncoef, S1[(i*(p0+1)+t)*dof*p0], false);
          for(Long k=0;k<dof*p0;k++){ // reverse permutation
            for(Long l=0;l<=p0;l++){
              for(Long j=0;j<(Long)coeff_perm[l].size();j++){
                Mout[k][coeff_perm[l][j]]=Bo[l][k][j];
              }
            }
          }
        }
      }
    }
  }
}

template <class Real> void SphericalHarmonics<Real>::RotateTranspose(const Vector<Real>& S_, Long p0, Long dof, Vector<Real>& S){
  std::vector<Matrix<Real>> Mr=MatRotate(p0);
  for(Long i=0;i<(Long)Mr.size();i++) Mr[i]=Mr[i].Transpose();
  std::vector<std::vector<Long>> coeff_perm(p0+1);
  { // Set coeff_perm
    for(Long n=0;n<=p0;n++) coeff_perm[n].resize(std::min(2*n+1,2*p0));
    Long itr=0;
    for(Long i=0;i<2*p0;i++){
      Long m=(i+1)/2;
      for(Long n=m;n<=p0;n++){
        coeff_perm[n][i]=itr;
        itr++;
      }
    }
  }
  Long Ncoef=p0*(p0+2);

  Long N=S_.Dim()/Ncoef/dof/(p0*(p0+1));
  assert(N*Ncoef*dof*(p0*(p0+1))==S_.Dim());
  if(S.Dim()!=N*dof*Ncoef*p0*(p0+1)) S.ReInit(N*dof*Ncoef*p0*(p0+1));
  Matrix<Real> S0(N*dof*p0*(p0+1),Ncoef, S.begin()                 , false);
  Matrix<Real> S1(N*dof*p0*(p0+1),Ncoef, (Iterator<Real>)S_.begin(), false);

  #pragma omp parallel
  { // Transpose all p0*(p0+1) rotations
    Integer tid=omp_get_thread_num();
    Integer omp_p=omp_get_num_threads();
    Matrix<Real> B0(dof*p0,Ncoef); // memory buffer

    std::vector<Matrix<Real>> Bi(p0+1), Bo(p0+1); // memory buffers
    for(Long i=0;i<=p0;i++){ // initialize Bi, Bo
      Bi[i].ReInit(dof*p0,coeff_perm[i].size());
      Bo[i].ReInit(dof*p0,coeff_perm[i].size());
    }

    Long a=(tid+0)*N/omp_p;
    Long b=(tid+1)*N/omp_p;
    for(Long i=a;i<b;i++){
      for(Long t=0;t<p0+1;t++){
        Long idx0=(i*(p0+1)+t)*p0*dof;
        { // Fast rotation
          Matrix<Real> Min(p0*dof,Ncoef, S1[idx0], false);
          for(Long k=0;k<dof*p0;k++){ // forward permutation
            for(Long l=0;l<=p0;l++){
              for(Long j=0;j<(Long)coeff_perm[l].size();j++){
                Bi[l][k][j]=Min[k][coeff_perm[l][j]];
              }
            }
          }
          for(Long l=0;l<=p0;l++){ // mat-vec
            Matrix<Real>::GEMM(Bo[l],Bi[l],Mr[t*(p0+1)+l]);
          }
          for(Long k=0;k<dof*p0;k++){ // reverse permutation
            for(Long l=0;l<=p0;l++){
              for(Long j=0;j<(Long)coeff_perm[l].size();j++){
                B0[k][coeff_perm[l][j]]=Bo[l][k][j];
              }
            }
          }
        }
        for(Long j=0;j<p0;j++){
          for(Long d=0;d<dof;d++){
            Long idx1=idx0+j*dof+d;
            Long offset=0;
            for(Long k=0;k<p0+1;k++){
              Real r[2]={cos(k*j*M_PI/p0),sin(k*j*M_PI/p0)}; // exp(i*k*theta)
              Long len=p0+1-k;
              if(k!=0 && k!=p0){
                for(Long l=0;l<len;l++){
                  Real x[2];
                  x[0]=B0[j*dof+d][offset+len*0+l];
                  x[1]=B0[j*dof+d][offset+len*1+l];
                  S0[idx1][offset+len*0+l]=x[0]*r[0]-x[1]*r[1];
                  S0[idx1][offset+len*1+l]=x[0]*r[1]+x[1]*r[0];
                }
                offset+=2*len;
              }else{
                for(Long l=0;l<len;l++){
                  S0[idx1][offset+l]=B0[j*dof+d][offset+l];
                }
                offset+=len;
              }
            }
            assert(offset==Ncoef);
          }
        }
      }
    }
  }
}

template <class Real> Vector<Real>& SphericalHarmonics<Real>::LegendreNodes(Long p1){
  assert(p1<SCTL_SHMAXDEG);
  assert(MatrixStore().Qx_.size() == SCTL_SHMAXDEG);
  Vector<Real>& Qx=MatrixStore().Qx_[p1];
  if(!Qx.Dim()){
    Vector<Real> qx1(p1+1);
    Vector<Real> qw1(p1+1);
    cgqf(p1+1, 1, 0.0, 0.0, -1.0, 1.0, &qx1[0], &qw1[0]);
    Qx=qx1;
  }
  return Qx;
}

template <class Real> Vector<Real>& SphericalHarmonics<Real>::LegendreWeights(Long p1){
  assert(p1<SCTL_SHMAXDEG);
  assert(MatrixStore().Qw_.size() == SCTL_SHMAXDEG);
  Vector<Real>& Qw=MatrixStore().Qw_[p1];
  if(!Qw.Dim()){
    // TODO: this works only for Real = double
    Vector<Real> qx1(p1+1);
    Vector<Real> qw1(p1+1);
    cgqf(p1+1, 1, 0.0, 0.0, -1.0, 1.0, &qx1[0], &qw1[0]);
    for(Long i=0;i<qw1.Dim();i++) qw1[i]*=M_PI/p1/sqrt(1-qx1[i]*qx1[i]);
    Qw=qw1;
  }
  return Qw;
}

template <class Real> Vector<Real>& SphericalHarmonics<Real>::SingularWeights(Long p1){
  assert(p1<SCTL_SHMAXDEG);
  assert(MatrixStore().Sw_.size() == SCTL_SHMAXDEG);
  Vector<Real>& Sw=MatrixStore().Sw_[p1];
  if(!Sw.Dim()){
    Vector<Real> qx1(p1+1);
    Vector<Real> qw1(p1+1);
    cgqf(p1+1, 1, 0.0, 0.0, -1.0, 1.0, &qx1[0], &qw1[0]);

    std::vector<Real> Yf(p1+1,0);
    { // Set Yf
      Real x0=1.0;
      std::vector<Real> alp0((p1+1)*(p1+2)/2);
      LegPoly(&alp0[0], &x0, 1, p1);

      std::vector<Real> alp((p1+1) * (p1+1)*(p1+2)/2);
      LegPoly(&alp[0], &qx1[0], p1+1, p1);

      for(Long j=0;j<p1+1;j++){
        for(Long i=0;i<p1+1;i++){
          Yf[i]+=4*M_PI/(2*j+1) * alp0[j] * alp[j*(p1+1)+i];
        }
      }
    }

    Sw.ReInit(p1+1);
    for(Long i=0;i<p1+1;i++){
      Sw[i]=(qw1[i]*M_PI/p1)*Yf[i]/cos(acos(qx1[i])/2);
    }
  }
  return Sw;
}

template <class Real> Matrix<Real>& SphericalHarmonics<Real>::MatFourier(Long p0, Long p1){
  assert(p0<SCTL_SHMAXDEG && p1<SCTL_SHMAXDEG);
  assert(MatrixStore().Mf_ .size() == SCTL_SHMAXDEG*SCTL_SHMAXDEG);
  Matrix<Real>& Mf =MatrixStore().Mf_ [p0*SCTL_SHMAXDEG+p1];
  if(!Mf.Dim(0)){
    const Real SQRT2PI=sqrt(2*M_PI);
    { // Set Mf
      Matrix<Real> M(2*p0,2*p1);
      for(Long j=0;j<2*p1;j++){
        M[0][j]=SQRT2PI*1.0;
        for(Long k=1;k<p0;k++){
          M[2*k-1][j]=SQRT2PI*cos(j*k*M_PI/p1);
          M[2*k-0][j]=SQRT2PI*sin(j*k*M_PI/p1);
        }
        M[2*p0-1][j]=SQRT2PI*cos(j*p0*M_PI/p1);
      }
      Mf=M;
    }
  }
  return Mf;
}

template <class Real> Matrix<Real>& SphericalHarmonics<Real>::MatFourierInv(Long p0, Long p1){
  assert(p0<SCTL_SHMAXDEG && p1<SCTL_SHMAXDEG);
  assert(MatrixStore().Mfinv_ .size() == SCTL_SHMAXDEG*SCTL_SHMAXDEG);
  Matrix<Real>& Mf =MatrixStore().Mfinv_ [p0*SCTL_SHMAXDEG+p1];
  if(!Mf.Dim(0)){
    const Real INVSQRT2PI=1.0/sqrt(2*M_PI)/p0;
    { // Set Mf
      Matrix<Real> M(2*p0,2*p1);
      M.SetZero();
      if(p1>p0) p1=p0;
      for(Long j=0;j<2*p0;j++){
        M[j][0]=INVSQRT2PI*0.5;
        for(Long k=1;k<p1;k++){
          M[j][2*k-1]=INVSQRT2PI*cos(j*k*M_PI/p0);
          M[j][2*k-0]=INVSQRT2PI*sin(j*k*M_PI/p0);
        }
        M[j][2*p1-1]=INVSQRT2PI*cos(j*p1*M_PI/p0);
      }
      if(p1==p0) for(Long j=0;j<2*p0;j++) M[j][2*p1-1]*=0.5;
      Mf=M;
    }
  }
  return Mf;
}

template <class Real> Matrix<Real>& SphericalHarmonics<Real>::MatFourierGrad(Long p0, Long p1){
  assert(p0<SCTL_SHMAXDEG && p1<SCTL_SHMAXDEG);
  assert(MatrixStore().Mdf_.size() == SCTL_SHMAXDEG*SCTL_SHMAXDEG);
  Matrix<Real>& Mdf=MatrixStore().Mdf_[p0*SCTL_SHMAXDEG+p1];
  if(!Mdf.Dim(0)){
    const Real SQRT2PI=sqrt(2*M_PI);
    { // Set Mdf_
      Matrix<Real> M(2*p0,2*p1);
      for(Long j=0;j<2*p1;j++){
        M[0][j]=SQRT2PI*0.0;
        for(Long k=1;k<p0;k++){
          M[2*k-1][j]=-SQRT2PI*k*sin(j*k*M_PI/p1);
          M[2*k-0][j]= SQRT2PI*k*cos(j*k*M_PI/p1);
        }
        M[2*p0-1][j]=-SQRT2PI*p0*sin(j*p0*M_PI/p1);
      }
      Mdf=M;
    }
  }
  return Mdf;
}

template <class Real> std::vector<Matrix<Real>>& SphericalHarmonics<Real>::MatLegendre(Long p0, Long p1){
  assert(p0<SCTL_SHMAXDEG && p1<SCTL_SHMAXDEG);
  assert(MatrixStore().Ml_ .size() == SCTL_SHMAXDEG*SCTL_SHMAXDEG);
  std::vector<Matrix<Real>>& Ml =MatrixStore().Ml_ [p0*SCTL_SHMAXDEG+p1];
  if(!Ml.size()){
    Vector<Real> qx1(p1+1);
    Vector<Real> qw1(p1+1);
    cgqf(p1+1, 1, 0.0, 0.0, -1.0, 1.0, &qx1[0], &qw1[0]);

    { // Set Ml
      Vector<Real> alp(qx1.Dim()*(p0+1)*(p0+2)/2);
      LegPoly(&alp[0], &qx1[0], qx1.Dim(), p0);

      Ml.resize(p0+1);
      auto ptr = alp.begin();
      for(Long i=0;i<=p0;i++){
        Ml[i].ReInit(p0+1-i, qx1.Dim(), ptr);
        ptr+=Ml[i].Dim(0)*Ml[i].Dim(1);
      }
    }
  }
  return Ml;
}

template <class Real> std::vector<Matrix<Real>>& SphericalHarmonics<Real>::MatLegendreInv(Long p0, Long p1){
  assert(p0<SCTL_SHMAXDEG && p1<SCTL_SHMAXDEG);
  assert(MatrixStore().Mlinv_ .size() == SCTL_SHMAXDEG*SCTL_SHMAXDEG);
  std::vector<Matrix<Real>>& Ml =MatrixStore().Mlinv_ [p0*SCTL_SHMAXDEG+p1];
  if(!Ml.size()){
    Vector<Real> qx1(p0+1);
    Vector<Real> qw1(p0+1);
    cgqf(p0+1, 1, 0.0, 0.0, -1.0, 1.0, &qx1[0], &qw1[0]);

    { // Set Ml
      Vector<Real> alp(qx1.Dim()*(p1+1)*(p1+2)/2);
      LegPoly(&alp[0], &qx1[0], qx1.Dim(), p1);

      Ml.resize(p1+1);
      auto ptr = alp.begin();
      for(Long i=0;i<=p1;i++){
        Ml[i].ReInit(qx1.Dim(), p1+1-i);
        Matrix<Real> M(p1+1-i, qx1.Dim(), ptr, false);
        for(Long j=0;j<p1+1-i;j++){ // Transpose and weights
          for(Long k=0;k<qx1.Dim();k++){
            Ml[i][k][j]=M[j][k]*qw1[k]*2*M_PI;
          }
        }
        ptr+=Ml[i].Dim(0)*Ml[i].Dim(1);
      }
    }
  }
  return Ml;
}

template <class Real> std::vector<Matrix<Real>>& SphericalHarmonics<Real>::MatLegendreGrad(Long p0, Long p1){
  assert(p0<SCTL_SHMAXDEG && p1<SCTL_SHMAXDEG);
  assert(MatrixStore().Mdl_.size() == SCTL_SHMAXDEG*SCTL_SHMAXDEG);
  std::vector<Matrix<Real>>& Mdl=MatrixStore().Mdl_[p0*SCTL_SHMAXDEG+p1];
  if(!Mdl.size()){
    Vector<Real> qx1(p1+1);
    Vector<Real> qw1(p1+1);
    cgqf(p1+1, 1, 0.0, 0.0, -1.0, 1.0, &qx1[0], &qw1[0]);

    { // Set Mdl
      Vector<Real> alp(qx1.Dim()*(p0+1)*(p0+2)/2);
      LegPolyDeriv(&alp[0], &qx1[0], qx1.Dim(), p0);

      Mdl.resize(p0+1);
      auto ptr = alp.begin();
      for(Long i=0;i<=p0;i++){
        Mdl[i].ReInit(p0+1-i, qx1.Dim(), ptr);
        ptr+=Mdl[i].Dim(0)*Mdl[i].Dim(1);
      }
    }
  }
  return Mdl;
}

template <class Real> std::vector<Matrix<Real>>& SphericalHarmonics<Real>::MatRotate(Long p0){
  std::vector<std::vector<Long>> coeff_perm(p0+1);
  { // Set coeff_perm
    for(Long n=0;n<=p0;n++) coeff_perm[n].resize(std::min(2*n+1,2*p0));
    Long itr=0;
    for(Long i=0;i<2*p0;i++){
      Long m=(i+1)/2;
      for(Long n=m;n<=p0;n++){
        coeff_perm[n][i]=itr;
        itr++;
      }
    }
  }

  assert(p0<SCTL_SHMAXDEG);
  assert(MatrixStore().Mr_.size() == SCTL_SHMAXDEG);
  std::vector<Matrix<Real>>& Mr=MatrixStore().Mr_[p0];
  if(!Mr.size()){
    const Real SQRT2PI=sqrt(2*M_PI);
    Long Ncoef=p0*(p0+2);
    Long Ngrid=2*p0*(p0+1);
    Long Naleg=(p0+1)*(p0+2)/2;

    Matrix<Real> Mcoord0(3,Ngrid);
    Vector<Real>& x=LegendreNodes(p0);
    for(Long i=0;i<p0+1;i++){ // Set Mcoord0
      for(Long j=0;j<2*p0;j++){
        Mcoord0[0][i*2*p0+j]=x[i];
        Mcoord0[1][i*2*p0+j]=sqrt(1-x[i]*x[i])*sin(M_PI*j/p0);
        Mcoord0[2][i*2*p0+j]=sqrt(1-x[i]*x[i])*cos(M_PI*j/p0);
      }
    }

    for(Long l=0;l<p0+1;l++){ // For each rotation angle
      Matrix<Real> Mcoord1;
      { // Rotate coordinates
        Matrix<Real> M(COORD_DIM, COORD_DIM);
        Real cos_=-x[l];
        Real sin_=-sqrt(1.0-x[l]*x[l]);
        M[0][0]= cos_; M[0][1]=0; M[0][2]=-sin_;
        M[1][0]=    0; M[1][1]=1; M[1][2]=    0;
        M[2][0]= sin_; M[2][1]=0; M[2][2]= cos_;
        Mcoord1=M*Mcoord0;
      }

      Matrix<Real> Mleg(Naleg, Ngrid);
      { // Set Mleg
        LegPoly(&Mleg[0][0], &Mcoord1[0][0], Ngrid, p0);
      }

      Vector<Real> theta(Ngrid);
      for(Long i=0;i<theta.Dim();i++){ // Set theta
        theta[i]=atan2(Mcoord1[1][i],Mcoord1[2][i]);
      }

      Matrix<Real> Mcoef2grid(Ncoef, Ngrid);
      { // Build Mcoef2grid
        Long offset0=0;
        Long offset1=0;
        for(Long i=0;i<p0+1;i++){
          Long len=p0+1-i;
          { // P * cos
            for(Long j=0;j<len;j++){
              for(Long k=0;k<Ngrid;k++){
                Mcoef2grid[offset1+j][k]=SQRT2PI*Mleg[offset0+j][k]*cos(i*theta[k]);
              }
            }
            offset1+=len;
          }
          if(i!=0 && i!=p0){ // P * sin
            for(Long j=0;j<len;j++){
              for(Long k=0;k<Ngrid;k++){
                Mcoef2grid[offset1+j][k]=SQRT2PI*Mleg[offset0+j][k]*sin(i*theta[k]);
              }
            }
            offset1+=len;
          }
          offset0+=len;
        }
        assert(offset0==Naleg);
        assert(offset1==Ncoef);
      }

      Vector<Real> Vcoef2coef(Ncoef*Ncoef);
      Vector<Real> Vcoef2grid(Ncoef*Ngrid, Mcoef2grid[0], false);
      Grid2SHC(Vcoef2grid, p0, p0, Vcoef2coef);

      Matrix<Real> Mcoef2coef(Ncoef, Ncoef, Vcoef2coef.begin(), false);
      for(Long n=0;n<=p0;n++){ // Create matrices for fast rotation
        Matrix<Real> M(coeff_perm[n].size(),coeff_perm[n].size());
        for(Long i=0;i<(Long)coeff_perm[n].size();i++){
          for(Long j=0;j<(Long)coeff_perm[n].size();j++){
            M[i][j]=Mcoef2coef[coeff_perm[n][i]][coeff_perm[n][j]];
          }
        }
        Mr.push_back(M);
      }
    }
  }
  return Mr;
}

template <class Real> void SphericalHarmonics<Real>::StokesSingularInteg(const Vector<Real>& S, Long p0, Long p1, Vector<Real>* SLMatrix, Vector<Real>* DLMatrix){
  Long Ngrid=2*p0*(p0+1);
  Long Ncoef=  p0*(p0+2);
  Long Nves=S.Dim()/(Ngrid*COORD_DIM);
  if(SLMatrix) SLMatrix->ReInit(Nves*(Ncoef*COORD_DIM)*(Ncoef*COORD_DIM));
  if(DLMatrix) DLMatrix->ReInit(Nves*(Ncoef*COORD_DIM)*(Ncoef*COORD_DIM));

  Long BLOCK_SIZE=(Long)6e9/((3*2*p1*(p1+1))*(3*2*p0*(p0+1))*2*8); // Limit memory usage to 6GB
  BLOCK_SIZE=std::min<Long>(BLOCK_SIZE,omp_get_max_threads());
  BLOCK_SIZE=std::max<Long>(BLOCK_SIZE,1);

  for(Long a=0;a<Nves;a+=BLOCK_SIZE){
    Long b=std::min(a+BLOCK_SIZE, Nves);

    Vector<Real> _SLMatrix, _DLMatrix, _S;
    if(SLMatrix) _SLMatrix.ReInit((b-a)*(Ncoef*COORD_DIM)*(Ncoef*COORD_DIM), SLMatrix->begin()+a*(Ncoef*COORD_DIM)*(Ncoef*COORD_DIM), false);
    if(DLMatrix) _DLMatrix.ReInit((b-a)*(Ncoef*COORD_DIM)*(Ncoef*COORD_DIM), DLMatrix->begin()+a*(Ncoef*COORD_DIM)*(Ncoef*COORD_DIM), false);
    _S                    .ReInit((b-a)*(Ngrid*COORD_DIM)                  , (Iterator<Real>)S.begin()+a*(Ngrid*COORD_DIM), false);

    if(SLMatrix && DLMatrix) StokesSingularInteg_< true,  true>(_S, p0, p1, _SLMatrix, _DLMatrix);
    else        if(SLMatrix) StokesSingularInteg_< true, false>(_S, p0, p1, _SLMatrix, _DLMatrix);
    else        if(DLMatrix) StokesSingularInteg_<false,  true>(_S, p0, p1, _SLMatrix, _DLMatrix);
  }
}

template <class Real> void SphericalHarmonics<Real>::LegPoly(Real* poly_val, const Real* X, Long N, Long degree){
  Real* p_val=poly_val;
  Real fact=1.0/(Real)sqrt(4*M_PI);

  std::vector<Real> u(N);
  for(Long n=0;n<N;n++){
    u[n]=sqrt(1-X[n]*X[n]);
    if(X[n]*X[n]>1.0) u[n]=0;
    p_val[n]=fact;
  }

  Real* p_val_nxt=poly_val;
  for(Long i=1;i<=degree;i++){
    p_val_nxt=&p_val_nxt[N*(degree-i+2)];
    Real c=(i==1?sqrt(3.0/2.0):1);
    if(i>1)c*=sqrt((Real)(2*i+1)/(2*i));
    for(Long n=0;n<N;n++){
      p_val_nxt[n]=-p_val[n]*u[n]*c;
    }
    p_val=p_val_nxt;
  }

  p_val=poly_val;
  for(Long m=0;m<degree;m++){
    for(Long n=0;n<N;n++){
      Real pmm=0;
      Real pmmp1=p_val[n];
      Real pll;
      for(Long ll=m+1;ll<=degree;ll++){
        Real a=sqrt(((Real)(2*ll-1)*(2*ll+1))/((ll-m)*(ll+m)));
        Real b=sqrt(((Real)(2*ll+1)*(ll+m-1)*(ll-m-1))/((ll-m)*(ll+m)*(2*ll-3)));
        pll=X[n]*a*pmmp1-b*pmm;
        pmm=pmmp1;
        pmmp1=pll;
        p_val[N*(ll-m)+n]=pll;
      }
    }
    p_val=&p_val[N*(degree-m+1)];
  }
}

template <class Real> void SphericalHarmonics<Real>::LegPolyDeriv(Real* poly_val, const Real* X, Long N, Long degree){
  std::vector<Real> leg_poly((degree+1)*(degree+2)*N/2);
  LegPoly(&leg_poly[0], X, N, degree);

  for(Long m=0;m<=degree;m++){
    for(Long n=0;n<=degree;n++) if(m<=n){
      const Real* Pn =&leg_poly[0];
      const Real* Pn_=&leg_poly[0];
      if((m+0)<=(n+0)) Pn =&leg_poly[N*(((degree*2-abs(m+0)+1)*abs(m+0))/2+(n+0))];
      if((m+1)<=(n+0)) Pn_=&leg_poly[N*(((degree*2-abs(m+1)+1)*abs(m+1))/2+(n+0))];
      Real*            Hn =&poly_val[N*(((degree*2-abs(m+0)+1)*abs(m+0))/2+(n+0))];

      Real c1=(abs(m+0)<=(n+0)?1.0:0)*m;
      Real c2=(abs(m+1)<=(n+0)?1.0:0)*sqrt(n+m+1)*sqrt(n>m?n-m:1);
      for(Long i=0;i<N;i++){
        Hn[i]=-(c1*X[i]*Pn[i]+c2*sqrt(1-X[i]*X[i])*Pn_[i])/sqrt(1-X[i]*X[i]);
      }
    }
  }
}

template <class Real> template <bool SLayer, bool DLayer> void SphericalHarmonics<Real>::StokesSingularInteg_(const Vector<Real>& X0, Long p0, Long p1, Vector<Real>& SL, Vector<Real>& DL){

  Profile::Tic("Rotate");
  static Vector<Real> S0, S;
  SphericalHarmonics<Real>::Grid2SHC(X0, p0, p0, S0);
  SphericalHarmonics<Real>::RotateAll(S0, p0, COORD_DIM, S);
  Profile::Toc();


  Profile::Tic("Upsample");
  Vector<Real> X, X_phi, X_theta, trg;
  SphericalHarmonics<Real>::SHC2Grid(S, p0, p1, X, &X_theta, &X_phi);
  SphericalHarmonics<Real>::SHC2Pole(S, p0, trg);
  Profile::Toc();


  Profile::Tic("Stokes");
  Vector<Real> SL0, DL0;
  { // Stokes kernel
    //Long M0=2*p0*(p0+1);
    Long M1=2*p1*(p1+1);
    Long N=trg.Dim()/(2*COORD_DIM);
    assert(X.Dim()==M1*COORD_DIM*N);
    if(SLayer && SL0.Dim()!=N*2*6*M1) SL0.ReInit(2*N*6*M1);
    if(DLayer && DL0.Dim()!=N*2*6*M1) DL0.ReInit(2*N*6*M1);
    Vector<Real>& qw=SphericalHarmonics<Real>::SingularWeights(p1);

    const Real scal_const_dl = 3.0/(4.0*M_PI);
    const Real scal_const_sl = 1.0/(8.0*M_PI);
    static Real eps=-1;
    if(eps<0){
      eps=1;
      while(eps*(Real)0.5+(Real)1.0>1.0) eps*=0.5;
    }

    #pragma omp parallel
    {
      Integer tid=omp_get_thread_num();
      Integer omp_p=omp_get_num_threads();

      Long a=(tid+0)*N/omp_p;
      Long b=(tid+1)*N/omp_p;
      for(Long i=a;i<b;i++){
        for(Long t=0;t<2;t++){
          Real tx, ty, tz;
          { // Read target coordinates
            tx=trg[i*2*COORD_DIM+0*2+t];
            ty=trg[i*2*COORD_DIM+1*2+t];
            tz=trg[i*2*COORD_DIM+2*2+t];
          }

          for(Long j0=0;j0<p1+1;j0++){
            for(Long j1=0;j1<2*p1;j1++){
              Long s=2*p1*j0+j1;

              Real dx, dy, dz;
              { // Compute dx, dy, dz
                dx=tx-X[(i*COORD_DIM+0)*M1+s];
                dy=ty-X[(i*COORD_DIM+1)*M1+s];
                dz=tz-X[(i*COORD_DIM+2)*M1+s];
              }

              Real nx, ny, nz;
              { // Compute source normal
                Real x_theta=X_theta[(i*COORD_DIM+0)*M1+s];
                Real y_theta=X_theta[(i*COORD_DIM+1)*M1+s];
                Real z_theta=X_theta[(i*COORD_DIM+2)*M1+s];

                Real x_phi=X_phi[(i*COORD_DIM+0)*M1+s];
                Real y_phi=X_phi[(i*COORD_DIM+1)*M1+s];
                Real z_phi=X_phi[(i*COORD_DIM+2)*M1+s];

                nx=(y_theta*z_phi-z_theta*y_phi);
                ny=(z_theta*x_phi-x_theta*z_phi);
                nz=(x_theta*y_phi-y_theta*x_phi);
              }

              Real area_elem=1.0;
              if(SLayer){ // Compute area_elem
                area_elem=sqrt(nx*nx+ny*ny+nz*nz);
              }

              Real rinv, rinv2;
              { // Compute rinv, rinv2
                Real r2=dx*dx+dy*dy+dz*dz;
                rinv=1.0/sqrt(r2);
                if(r2<=eps) rinv=0;
                rinv2=rinv*rinv;
              }

              if(DLayer){
                Real rinv5=rinv2*rinv2*rinv;
                Real r_dot_n_rinv5=scal_const_dl*qw[j0*t+(p1-j0)*(1-t)] * (nx*dx+ny*dy+nz*dz)*rinv5;
                DL0[((i*2+t)*6+0)*M1+s]=dx*dx*r_dot_n_rinv5;
                DL0[((i*2+t)*6+1)*M1+s]=dx*dy*r_dot_n_rinv5;
                DL0[((i*2+t)*6+2)*M1+s]=dx*dz*r_dot_n_rinv5;
                DL0[((i*2+t)*6+3)*M1+s]=dy*dy*r_dot_n_rinv5;
                DL0[((i*2+t)*6+4)*M1+s]=dy*dz*r_dot_n_rinv5;
                DL0[((i*2+t)*6+5)*M1+s]=dz*dz*r_dot_n_rinv5;
              }
              if(SLayer){
                Real area_rinv =scal_const_sl*qw[j0*t+(p1-j0)*(1-t)] * area_elem*rinv;
                Real area_rinv2=area_rinv*rinv2;
                SL0[((i*2+t)*6+0)*M1+s]=area_rinv+dx*dx*area_rinv2;
                SL0[((i*2+t)*6+1)*M1+s]=          dx*dy*area_rinv2;
                SL0[((i*2+t)*6+2)*M1+s]=          dx*dz*area_rinv2;
                SL0[((i*2+t)*6+3)*M1+s]=area_rinv+dy*dy*area_rinv2;
                SL0[((i*2+t)*6+4)*M1+s]=          dy*dz*area_rinv2;
                SL0[((i*2+t)*6+5)*M1+s]=area_rinv+dz*dz*area_rinv2;
              }
            }
          }
        }
      }
    }
    Profile::Add_FLOP(20*(2*p1)*(p1+1)*2*N);
    if(SLayer) Profile::Add_FLOP((19+6)*(2*p1)*(p1+1)*2*N);
    if(DLayer) Profile::Add_FLOP( 22   *(2*p1)*(p1+1)*2*N);
  }
  Profile::Toc();


  Profile::Tic("UpsampleTranspose");
  static Vector<Real> SL1, DL1;
  SphericalHarmonics<Real>::SHC2GridTranspose(SL0, p1, p0, SL1);
  SphericalHarmonics<Real>::SHC2GridTranspose(DL0, p1, p0, DL1);
  Profile::Toc();


  Profile::Tic("RotateTranspose");
  static Vector<Real> SL2, DL2;
  SphericalHarmonics<Real>::RotateTranspose(SL1, p0, 2*6, SL2);
  SphericalHarmonics<Real>::RotateTranspose(DL1, p0, 2*6, DL2);
  Profile::Toc();


  Profile::Tic("Rearrange");
  static Vector<Real> SL3, DL3;
  { // Transpose
    Long Ncoef=p0*(p0+2);
    Long Ngrid=2*p0*(p0+1);
    { // Transpose SL2
      Long N=SL2.Dim()/(6*Ncoef*Ngrid);
      SL3.ReInit(N*COORD_DIM*Ncoef*COORD_DIM*Ngrid);
      #pragma omp parallel
      {
        Integer tid=omp_get_thread_num();
        Integer omp_p=omp_get_num_threads();
        Matrix<Real> B(COORD_DIM*Ncoef,Ngrid*COORD_DIM);

        Long a=(tid+0)*N/omp_p;
        Long b=(tid+1)*N/omp_p;
        for(Long i=a;i<b;i++){
          Matrix<Real> M0(Ngrid*6, Ncoef, SL2.begin()+i*Ngrid*6*Ncoef, false);
          for(Long k=0;k<Ncoef;k++){ // Transpose
            for(Long j=0;j<Ngrid;j++){ // TODO: needs blocking
              B[k+Ncoef*0][j*COORD_DIM+0]=M0[j*6+0][k];
              B[k+Ncoef*1][j*COORD_DIM+0]=M0[j*6+1][k];
              B[k+Ncoef*2][j*COORD_DIM+0]=M0[j*6+2][k];
              B[k+Ncoef*0][j*COORD_DIM+1]=M0[j*6+1][k];
              B[k+Ncoef*1][j*COORD_DIM+1]=M0[j*6+3][k];
              B[k+Ncoef*2][j*COORD_DIM+1]=M0[j*6+4][k];
              B[k+Ncoef*0][j*COORD_DIM+2]=M0[j*6+2][k];
              B[k+Ncoef*1][j*COORD_DIM+2]=M0[j*6+4][k];
              B[k+Ncoef*2][j*COORD_DIM+2]=M0[j*6+5][k];
            }
          }
          Matrix<Real> M1(Ncoef*COORD_DIM, COORD_DIM*Ngrid, SL3.begin()+i*COORD_DIM*Ncoef*COORD_DIM*Ngrid, false);
          for(Long k=0;k<B.Dim(0);k++){ // Rearrange
            for(Long j0=0;j0<COORD_DIM;j0++){
              for(Long j1=0;j1<p0+1;j1++){
                for(Long j2=0;j2<p0;j2++) M1[k][((j0*(p0+1)+   j1)*2+0)*p0+j2]=B[k][((j1*p0+j2)*2+0)*COORD_DIM+j0];
                for(Long j2=0;j2<p0;j2++) M1[k][((j0*(p0+1)+p0-j1)*2+1)*p0+j2]=B[k][((j1*p0+j2)*2+1)*COORD_DIM+j0];
              }
            }
          }
        }
      }
    }
    { // Transpose DL2
      Long N=DL2.Dim()/(6*Ncoef*Ngrid);
      DL3.ReInit(N*COORD_DIM*Ncoef*COORD_DIM*Ngrid);
      #pragma omp parallel
      {
        Integer tid=omp_get_thread_num();
        Integer omp_p=omp_get_num_threads();
        Matrix<Real> B(COORD_DIM*Ncoef,Ngrid*COORD_DIM);

        Long a=(tid+0)*N/omp_p;
        Long b=(tid+1)*N/omp_p;
        for(Long i=a;i<b;i++){
          Matrix<Real> M0(Ngrid*6, Ncoef, DL2.begin()+i*Ngrid*6*Ncoef, false);
          for(Long k=0;k<Ncoef;k++){ // Transpose
            for(Long j=0;j<Ngrid;j++){ // TODO: needs blocking
              B[k+Ncoef*0][j*COORD_DIM+0]=M0[j*6+0][k];
              B[k+Ncoef*1][j*COORD_DIM+0]=M0[j*6+1][k];
              B[k+Ncoef*2][j*COORD_DIM+0]=M0[j*6+2][k];
              B[k+Ncoef*0][j*COORD_DIM+1]=M0[j*6+1][k];
              B[k+Ncoef*1][j*COORD_DIM+1]=M0[j*6+3][k];
              B[k+Ncoef*2][j*COORD_DIM+1]=M0[j*6+4][k];
              B[k+Ncoef*0][j*COORD_DIM+2]=M0[j*6+2][k];
              B[k+Ncoef*1][j*COORD_DIM+2]=M0[j*6+4][k];
              B[k+Ncoef*2][j*COORD_DIM+2]=M0[j*6+5][k];
            }
          }
          Matrix<Real> M1(Ncoef*COORD_DIM, COORD_DIM*Ngrid, DL3.begin()+i*COORD_DIM*Ncoef*COORD_DIM*Ngrid, false);
          for(Long k=0;k<B.Dim(0);k++){ // Rearrange
            for(Long j0=0;j0<COORD_DIM;j0++){
              for(Long j1=0;j1<p0+1;j1++){
                for(Long j2=0;j2<p0;j2++) M1[k][((j0*(p0+1)+   j1)*2+0)*p0+j2]=B[k][((j1*p0+j2)*2+0)*COORD_DIM+j0];
                for(Long j2=0;j2<p0;j2++) M1[k][((j0*(p0+1)+p0-j1)*2+1)*p0+j2]=B[k][((j1*p0+j2)*2+1)*COORD_DIM+j0];
              }
            }
          }
        }
      }
    }
  }
  Profile::Toc();


  Profile::Tic("Grid2SHC");
  SphericalHarmonics<Real>::Grid2SHC(SL3, p0, p0, SL);
  SphericalHarmonics<Real>::Grid2SHC(DL3, p0, p0, DL);
  Profile::Toc();

}

template <class Real> void SphericalHarmonics<Real>::WriteVTK(const char* fname, long p0, long p1, Real period, const Vector<Real>* S, const Vector<Real>* v_ptr, MPI_Comm comm){
  typedef double VTKReal;

  Vector<Real> SS;
  if (S == nullptr) {
    Integer p = 2;
    Integer Ncoeff = p * (p + 2);
    Vector<Real> SSS(COORD_DIM * Ncoeff);
    SSS.SetZero();
    SSS[1+0*p+0*Ncoeff] = sqrt<Real>(2.0)/sqrt<Real>(3.0);
    SSS[1+1*p+1*Ncoeff] = 2/sqrt<Real>(3.0);
    SSS[1+2*p+2*Ncoeff] = 2/sqrt<Real>(3.0);
    SphericalHarmonics<Real>::SHC2Grid(SSS, p, p0, SS);
    S = &SS;
  }

  Vector<Real> X, Xp, V, Vp;
  { // Upsample X
    const Vector<Real>& X0=*S;
    Vector<Real> X1;
    SphericalHarmonics<Real>::Grid2SHC(X0, p0, p0, X1);
    SphericalHarmonics<Real>::SHC2Grid(X1, p0, p1, X);
    SphericalHarmonics<Real>::SHC2Pole(X1, p0,     Xp);
  }
  if(v_ptr){ // Upsample V
    const Vector<Real>& X0=*v_ptr;
    Vector<Real> X1;
    SphericalHarmonics<Real>::Grid2SHC(X0, p0, p0, X1);
    SphericalHarmonics<Real>::SHC2Grid(X1, p0, p1, V );
    SphericalHarmonics<Real>::SHC2Pole(X1, p0,     Vp);
  }

  std::vector<VTKReal> point_coord;
  std::vector<VTKReal> point_value;
  std::vector<int32_t> poly_connect;
  std::vector<int32_t> poly_offset;
  { // Set point_coord, point_value, poly_connect
    Long N_ves = X.Dim()/(2*p1*(p1+1)*COORD_DIM); // Number of vesicles
    assert(Xp.Dim() == N_ves*2*COORD_DIM);
    for(Long k=0;k<N_ves;k++){ // Set point_coord
      Real C[COORD_DIM]={0,0,0};
      if(period>0){
        for(Integer l=0;l<COORD_DIM;l++) C[l]=0;
        for(Long i=0;i<p1+1;i++){
          for(Long j=0;j<2*p1;j++){
            for(Integer l=0;l<COORD_DIM;l++){
              C[l]+=X[j+2*p1*(i+(p1+1)*(l+k*COORD_DIM))];
            }
          }
        }
        for(Integer l=0;l<COORD_DIM;l++) C[l]+=Xp[0+2*(l+k*COORD_DIM)];
        for(Integer l=0;l<COORD_DIM;l++) C[l]+=Xp[1+2*(l+k*COORD_DIM)];
        for(Integer l=0;l<COORD_DIM;l++) C[l]/=2*p1*(p1+1)+2;
        for(Integer l=0;l<COORD_DIM;l++) C[l]=(round(C[l]/period))*period;
      }

      for(Long i=0;i<p1+1;i++){
        for(Long j=0;j<2*p1;j++){
          for(Integer l=0;l<COORD_DIM;l++){
            point_coord.push_back(X[j+2*p1*(i+(p1+1)*(l+k*COORD_DIM))]-C[l]);
          }
        }
      }
      for(Integer l=0;l<COORD_DIM;l++) point_coord.push_back(Xp[0+2*(l+k*COORD_DIM)]-C[l]);
      for(Integer l=0;l<COORD_DIM;l++) point_coord.push_back(Xp[1+2*(l+k*COORD_DIM)]-C[l]);
    }

    if(v_ptr) {
      Long data__dof = V.Dim() / (2*p1*(p1+1));
      for(Long k=0;k<N_ves;k++){ // Set point_value
        for(Long i=0;i<p1+1;i++){
          for(Long j=0;j<2*p1;j++){
            for(Long l=0;l<data__dof;l++){
              point_value.push_back(V[j+2*p1*(i+(p1+1)*(l+k*data__dof))]);
            }
          }
        }
        for(Long l=0;l<data__dof;l++) point_value.push_back(Vp[0+2*(l+k*data__dof)]);
        for(Long l=0;l<data__dof;l++) point_value.push_back(Vp[1+2*(l+k*data__dof)]);
      }
    }

    for(Long k=0;k<N_ves;k++){
      for(Long j=0;j<2*p1;j++){
        Long i0= 0;
        Long i1=p1;
        Long j0=((j+0)       );
        Long j1=((j+1)%(2*p1));

        poly_connect.push_back((2*p1*(p1+1)+2)*k + 2*p1*(p1+1)+0);
        poly_connect.push_back((2*p1*(p1+1)+2)*k + 2*p1*i0+j0);
        poly_connect.push_back((2*p1*(p1+1)+2)*k + 2*p1*i0+j1);
        poly_offset.push_back(poly_connect.size());

        poly_connect.push_back((2*p1*(p1+1)+2)*k + 2*p1*(p1+1)+1);
        poly_connect.push_back((2*p1*(p1+1)+2)*k + 2*p1*i1+j0);
        poly_connect.push_back((2*p1*(p1+1)+2)*k + 2*p1*i1+j1);
        poly_offset.push_back(poly_connect.size());
      }
      for(Long i=0;i<p1;i++){
        for(Long j=0;j<2*p1;j++){
          Long i0=((i+0)       );
          Long i1=((i+1)       );
          Long j0=((j+0)       );
          Long j1=((j+1)%(2*p1));
          poly_connect.push_back((2*p1*(p1+1)+2)*k + 2*p1*i0+j0);
          poly_connect.push_back((2*p1*(p1+1)+2)*k + 2*p1*i1+j0);
          poly_connect.push_back((2*p1*(p1+1)+2)*k + 2*p1*i1+j1);
          poly_connect.push_back((2*p1*(p1+1)+2)*k + 2*p1*i0+j1);
          poly_offset.push_back(poly_connect.size());
        }
      }
    }
  }

  int myrank, np;
  MPI_Comm_size(comm,&np);
  MPI_Comm_rank(comm,&myrank);

  std::vector<VTKReal>& coord=point_coord;
  std::vector<VTKReal>& value=point_value;
  std::vector<int32_t>& connect=poly_connect;
  std::vector<int32_t>& offset=poly_offset;

  Long pt_cnt=coord.size()/COORD_DIM;
  Long poly_cnt=poly_offset.size();

  // Open file for writing.
  std::stringstream vtufname;
  vtufname<<fname<<"_"<<std::setfill('0')<<std::setw(6)<<myrank<<".vtp";
  std::ofstream vtufile;
  vtufile.open(vtufname.str().c_str());
  if(vtufile.fail()) return;

  bool isLittleEndian;
  { // Set isLittleEndian
    uint16_t number = 0x1;
    uint8_t *numPtr = (uint8_t*)&number;
    isLittleEndian=(numPtr[0] == 1);
  }

  // Proceed to write to file.
  Long data_size=0;
  vtufile<<"<?xml version=\"1.0\"?>\n";
  if(isLittleEndian) vtufile<<"<VTKFile type=\"PolyData\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
  else               vtufile<<"<VTKFile type=\"PolyData\" version=\"0.1\" byte_order=\"BigEndian\">\n";
  //===========================================================================
  vtufile<<"  <PolyData>\n";
  vtufile<<"    <Piece NumberOfPoints=\""<<pt_cnt<<"\" NumberOfVerts=\"0\" NumberOfLines=\"0\" NumberOfStrips=\"0\" NumberOfPolys=\""<<poly_cnt<<"\">\n";

  //---------------------------------------------------------------------------
  vtufile<<"      <Points>\n";
  vtufile<<"        <DataArray type=\"Float"<<sizeof(VTKReal)*8<<"\" NumberOfComponents=\""<<COORD_DIM<<"\" Name=\"Position\" format=\"appended\" offset=\""<<data_size<<"\" />\n";
  data_size+=sizeof(uint32_t)+coord.size()*sizeof(VTKReal);
  vtufile<<"      </Points>\n";
  //---------------------------------------------------------------------------
  if(value.size()){ // value
    vtufile<<"      <PointData>\n";
    vtufile<<"        <DataArray type=\"Float"<<sizeof(VTKReal)*8<<"\" NumberOfComponents=\""<<value.size()/pt_cnt<<"\" Name=\""<<"value"<<"\" format=\"appended\" offset=\""<<data_size<<"\" />\n";
    data_size+=sizeof(uint32_t)+value.size()*sizeof(VTKReal);
    vtufile<<"      </PointData>\n";
  }
  //---------------------------------------------------------------------------
  vtufile<<"      <Polys>\n";
  vtufile<<"        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"appended\" offset=\""<<data_size<<"\" />\n";
  data_size+=sizeof(uint32_t)+connect.size()*sizeof(int32_t);
  vtufile<<"        <DataArray type=\"Int32\" Name=\"offsets\" format=\"appended\" offset=\""<<data_size<<"\" />\n";
  data_size+=sizeof(uint32_t)+offset.size() *sizeof(int32_t);
  vtufile<<"      </Polys>\n";
  //---------------------------------------------------------------------------

  vtufile<<"    </Piece>\n";
  vtufile<<"  </PolyData>\n";
  //===========================================================================
  vtufile<<"  <AppendedData encoding=\"raw\">\n";
  vtufile<<"    _";

  int32_t block_size;
  block_size=coord.size()*sizeof(VTKReal); vtufile.write((char*)&block_size, sizeof(int32_t)); vtufile.write((char*)&coord  [0], coord.size()*sizeof(VTKReal));
  if(value.size()){ // value
    block_size=value.size()*sizeof(VTKReal); vtufile.write((char*)&block_size, sizeof(int32_t)); vtufile.write((char*)&value  [0], value.size()*sizeof(VTKReal));
  }
  block_size=connect.size()*sizeof(int32_t); vtufile.write((char*)&block_size, sizeof(int32_t)); vtufile.write((char*)&connect[0], connect.size()*sizeof(int32_t));
  block_size=offset .size()*sizeof(int32_t); vtufile.write((char*)&block_size, sizeof(int32_t)); vtufile.write((char*)&offset [0], offset .size()*sizeof(int32_t));

  vtufile<<"\n";
  vtufile<<"  </AppendedData>\n";
  //===========================================================================
  vtufile<<"</VTKFile>\n";
  vtufile.close();


  if(myrank) return;
  std::stringstream pvtufname;
  pvtufname<<fname<<".pvtp";
  std::ofstream pvtufile;
  pvtufile.open(pvtufname.str().c_str());
  if(pvtufile.fail()) return;
  pvtufile<<"<?xml version=\"1.0\"?>\n";
  pvtufile<<"<VTKFile type=\"PPolyData\">\n";
  pvtufile<<"  <PPolyData GhostLevel=\"0\">\n";
  pvtufile<<"      <PPoints>\n";
  pvtufile<<"        <PDataArray type=\"Float"<<sizeof(VTKReal)*8<<"\" NumberOfComponents=\""<<COORD_DIM<<"\" Name=\"Position\"/>\n";
  pvtufile<<"      </PPoints>\n";
  if(value.size()){ // value
    pvtufile<<"      <PPointData>\n";
    pvtufile<<"        <PDataArray type=\"Float"<<sizeof(VTKReal)*8<<"\" NumberOfComponents=\""<<value.size()/pt_cnt<<"\" Name=\""<<"value"<<"\"/>\n";
    pvtufile<<"      </PPointData>\n";
  }
  {
    // Extract filename from path.
    std::stringstream vtupath;
    vtupath<<'/'<<fname;
    std::string pathname = vtupath.str();
    auto found = pathname.find_last_of("/\\");
    std::string fname_ = pathname.substr(found+1);
    for(Integer i=0;i<np;i++) pvtufile<<"      <Piece Source=\""<<fname_<<"_"<<std::setfill('0')<<std::setw(6)<<i<<".vtp\"/>\n";
  }
  pvtufile<<"  </PPolyData>\n";
  pvtufile<<"</VTKFile>\n";
  pvtufile.close();
}

}  // end namespace
