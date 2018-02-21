#include SCTL_INCLUDE(legendre_rule.hpp)

// TODO: Replace work vectors with dynamic-arrays

namespace SCTL_NAMESPACE {

template <class Real> void SphericalHarmonics<Real>::Grid2SHC(const Vector<Real>& X, Long Nt, Long Np, Long p1, Vector<Real>& S, SHCArrange arrange){
  Long N = X.Dim() / (Np*Nt);
  assert(X.Dim() == N*Np*Nt);

  Vector<Real> B1(N*(p1+1)*(p1+1));
  Grid2SHC_(X, Nt, Np, p1, B1);
  SHCArrange0(B1, p1, S, arrange);
}

template <class Real> void SphericalHarmonics<Real>::SHC2Grid(const Vector<Real>& S, SHCArrange arrange, Long p0, Long Nt, Long Np, Vector<Real>* X, Vector<Real>* X_theta, Vector<Real>* X_phi){
  Vector<Real> B0;
  SHCArrange1(S, arrange, p0, B0);
  SHC2Grid_(B0, p0, Nt, Np, X, X_phi, X_theta);
}

template <class Real> void SphericalHarmonics<Real>::SHCEval(const Vector<Real>& S, SHCArrange arrange, Long p0, const Vector<Real>& cos_theta_phi, Vector<Real>& X) {
  Long M = (p0+1) * (p0+1);

  Long dof;
  Matrix<Real> B1;
  { // Set B1, dof
    Vector<Real> B0;
    SHCArrange1(S, arrange, p0, B0);
    dof = B0.Dim() / M;
    assert(B0.Dim() == dof * M);

    B1.ReInit(dof, M);
    Vector<Real> B1_(B1.Dim(0) * B1.Dim(1), B1.begin(), false);
    SHCArrange0(B0, p0, B1_, SHCArrange::COL_MAJOR_NONZERO);
  }
  assert(B1.Dim(0) == dof);
  assert(B1.Dim(1) == M);

  Matrix<Real> SHBasis;
  SHBasisEval(p0, cos_theta_phi, SHBasis);
  assert(SHBasis.Dim(1) == M);
  Long N = SHBasis.Dim(0);

  { // Set X
    if (X.Dim() != N*dof) X.ReInit(N * dof);
    for (Long k0 = 0; k0 < N; k0++) {
      for (Long k1 = 0; k1 < dof; k1++) {
        Real X_ = 0;
        for (Long i = 0; i < M; i++) X_ += B1[k1][i] * SHBasis[k0][i];
        X[k0 * dof + k1] = X_;
      }
    }
  }
}

template <class Real> void SphericalHarmonics<Real>::SHC2Pole(const Vector<Real>& S, SHCArrange arrange, Long p0, Vector<Real>& P){
  Vector<Real> QP[2];
  { // Set QP // TODO: store these weights
    Vector<Real> x(1), alp;
    const Real SQRT2PI = sqrt<Real>(4 * const_pi<Real>());
    for (Long i = 0; i < 2; i++) {
      x = (i ? -1 : 1);
      LegPoly(alp, x, p0);
      QP[i].ReInit(p0 + 1, alp.begin());
      QP[i] *= SQRT2PI;
    }
  }

  Long M, N;
  { // Set M, N
    M = 0;
    if (arrange == SHCArrange::ALL) M = 2*(p0+1)*(p0+1);
    if (arrange == SHCArrange::ROW_MAJOR) M = (p0+1)*(p0+2);
    if (arrange == SHCArrange::COL_MAJOR_NONZERO) M = (p0+1)*(p0+1);
    if (M == 0) return;
    N = S.Dim() / M;
    assert(S.Dim() == N * M);
  }
  if(P.Dim() != N * 2) P.ReInit(N * 2);

  if (arrange == SHCArrange::ALL) {
    #pragma omp parallel
    { // Compute pole
      Integer tid = omp_get_thread_num();
      Integer omp_p = omp_get_num_threads();

      Long a = (tid + 0) * N / omp_p;
      Long b = (tid + 1) * N / omp_p;
      for (Long i = a; i < b; i++) {
        Real P_[2] = {0, 0};
        for (Long j = 0; j < p0 + 1; j++) {
          P_[0] += S[i*M + j*(p0+1)*2] * QP[0][j];
          P_[1] += S[i*M + j*(p0+1)*2] * QP[1][j];
        }
        P[2*i+0] = P_[0];
        P[2*i+1] = P_[1];
      }
    }
  }
  if (arrange == SHCArrange::ROW_MAJOR) {
    #pragma omp parallel
    { // Compute pole
      Integer tid = omp_get_thread_num();
      Integer omp_p = omp_get_num_threads();

      Long a = (tid + 0) * N / omp_p;
      Long b = (tid + 1) * N / omp_p;
      for (Long i = a; i < b; i++) {
        Long idx = 0;
        Real P_[2] = {0, 0};
        for (Long j = 0; j < p0 + 1; j++) {
          P_[0] += S[i*M+idx] * QP[0][j];
          P_[1] += S[i*M+idx] * QP[1][j];
          idx += 2*(j+1);
        }
        P[2*i+0] = P_[0];
        P[2*i+1] = P_[1];
      }
    }
  }
  if (arrange == SHCArrange::COL_MAJOR_NONZERO) {
    #pragma omp parallel
    { // Compute pole
      Integer tid = omp_get_thread_num();
      Integer omp_p = omp_get_num_threads();

      Long a = (tid + 0) * N / omp_p;
      Long b = (tid + 1) * N / omp_p;
      for (Long i = a; i < b; i++) {
        Real P_[2] = {0, 0};
        for (Long j = 0; j < p0 + 1; j++) {
          P_[0] += S[i*M+j] * QP[0][j];
          P_[1] += S[i*M+j] * QP[1][j];
        }
        P[2*i+0] = P_[0];
        P[2*i+1] = P_[1];
      }
    }
  }
}

template <class Real> void SphericalHarmonics<Real>::WriteVTK(const char* fname, const Vector<Real>* S, const Vector<Real>* v_ptr, SHCArrange arrange, Long p0, Long p1, Real period, const Comm& comm){
  typedef double VTKReal;

  Vector<Real> SS;
  if (S == nullptr) {
    Integer p = 2;
    Integer Ncoeff = (p + 1) * (p + 1);
    Vector<Real> SSS(COORD_DIM * Ncoeff), SSS_grid;
    SSS.SetZero();
    SSS[1+0*p+0*Ncoeff] = sqrt<Real>(2.0)/sqrt<Real>(3.0);
    SSS[1+1*p+1*Ncoeff] = 1/sqrt<Real>(3.0);
    SSS[1+2*p+2*Ncoeff] = 1/sqrt<Real>(3.0);
    SphericalHarmonics<Real>::SHC2Grid(SSS, SHCArrange::COL_MAJOR_NONZERO, p, p+1, 2*p+2, &SSS_grid);
    SphericalHarmonics<Real>::Grid2SHC(SSS_grid, p+1, 2*p+2, p0, SS, arrange);
    S = &SS;
  }

  Vector<Real> X, Xp, V, Vp;
  { // Upsample X
    const Vector<Real>& X0=*S;
    SphericalHarmonics<Real>::SHC2Grid(X0, arrange, p0, p1+1, 2*p1, &X);
    SphericalHarmonics<Real>::SHC2Pole(X0, arrange, p0, Xp);
  }
  if(v_ptr){ // Upsample V
    const Vector<Real>& X0=*v_ptr;
    SphericalHarmonics<Real>::SHC2Grid(X0, arrange, p0, p1+1, 2*p1, &V);
    SphericalHarmonics<Real>::SHC2Pole(X0, arrange, p0, Vp);
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

  Integer np = comm.Size();
  Integer myrank = comm.Rank();

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


template <class Real> void SphericalHarmonics<Real>::Grid2VecSHC(const Vector<Real>& X, Long Nt, Long Np, Long p0, Vector<Real>& S, SHCArrange arrange) {
  Long N = X.Dim() / (Np*Nt);
  assert(X.Dim() == N*Np*Nt);
  assert(N % COORD_DIM == 0);

  Vector<Real> B0(N*Nt*Np);
  { // Set B0
    Vector<Real> sin_phi(Np), cos_phi(Np);
    for (Long i = 0; i < Np; i++) {
      sin_phi[i] = sin(2 * const_pi<Real>() * i / Np);
      cos_phi[i] = cos(2 * const_pi<Real>() * i / Np);
    }
    const auto& Y = LegendreNodes(Nt - 1);
    assert(Y.Dim() == Nt);
    Long Ngrid = Nt * Np;
    for (Long k = 0; k < N; k+=COORD_DIM) {
      for (Long i = 0; i < Nt; i++) {
        Real sin_theta = sqrt<Real>(1 - Y[i]*Y[i]);
        Real cos_theta = Y[i];
        Real s = 1 / sin_theta;
        const auto X_ = X.begin() + (k*Nt+i)*Np;
        auto B0_ = B0.begin() + (k*Nt+i)*Np;
        for (Long j = 0; j < Np; j++) {
          StaticArray<Real,3> in;
          in[0] = X_[0*Ngrid+j];
          in[1] = X_[1*Ngrid+j];
          in[2] = X_[2*Ngrid+j];

          StaticArray<Real,9> M;
          M[0] = sin_theta*cos_phi[j]; M[1] = sin_theta*sin_phi[j]; M[2] = cos_theta;
          M[3] = cos_theta*cos_phi[j]; M[4] = cos_theta*sin_phi[j]; M[5] =-sin_theta;
          M[6] =          -sin_phi[j]; M[7] =           cos_phi[j]; M[8] =         0;

          B0_[0*Ngrid+j] = ( M[0] * in[0] + M[1] * in[1] + M[2] * in[2] );
          B0_[1*Ngrid+j] = ( M[3] * in[0] + M[4] * in[1] + M[5] * in[2] ) * s;
          B0_[2*Ngrid+j] = ( M[6] * in[0] + M[7] * in[1] + M[8] * in[2] ) * s;
        }
      }
    }
  }

  Long p_ = p0 + 1;
  Long M0 = (p0+1)*(p0+1);
  Long M_ = (p_+1)*(p_+1);
  Vector<Real> B1(N*M_);
  Grid2SHC_(B0, Nt, Np, p_, B1);

  Vector<Real> B2(N*M0);
  const Complex<Real> imag(0,1);
  for (Long i=0; i<N; i+=COORD_DIM) {
    for (Long m=0; m<=p0; m++) {
      for (Long n=m; n<=p0; n++) {
        auto read_coeff = [&](const Vector<Real>& coeff, Long i, Long p, Long n, Long m) {
          Complex<Real> c;
          if (0<=m && m<=n && n<=p) {
            Long idx_real = ((2*p-m+3)*m - (m?p+1:0))*N + (p+1-m)*i - m + n;
            Long idx_imag = idx_real + (p+1-m)*N;
            c.real = coeff[idx_real];
            if (m) c.imag = coeff[idx_imag];
          }
          return c;
        };
        auto write_coeff = [&](Complex<Real> c, Vector<Real>& coeff, Long i, Long p, Long n, Long m) {
          if (0<=m && m<=n && n<=p) {
            Long idx_real = ((2*p-m+3)*m - (m?p+1:0))*N + (p+1-m)*i - m + n;
            Long idx_imag = idx_real + (p+1-m)*N;
            coeff[idx_real] = c.real;
            if (m) coeff[idx_imag] = c.imag;
          }
        };

        auto gr = [&](Long n, Long m) { return read_coeff(B1, i+0, p_, n, m); };
        auto gt = [&](Long n, Long m) { return read_coeff(B1, i+1, p_, n, m); };
        auto gp = [&](Long n, Long m) { return read_coeff(B1, i+2, p_, n, m); };

        Complex<Real> phiY, phiG, phiX;
        { // (phiG, phiX) <-- (gt, gp)
          auto A = [&](Long n, Long m) { return (0<=n && m<=n && n<=p_ ? sqrt<Real>(n*n * ((n+1)*(n+1) - m*m) / (Real)((2*n+1)*(2*n+3))) : 0); };
          auto B = [&](Long n, Long m) { return (0<=n && m<=n && n<=p_ ? sqrt<Real>((n+1)*(n+1) * (n*n - m*m) / (Real)((2*n+1)*(2*n-1))) : 0); };
          phiY = gr(n,m);
          phiG = (gt(n+1,m)*A(n,m) - gt(n-1,m)*B(n,m) - imag*m*gp(n,m)) * (1/(Real)(std::max<Long>(n,1)*(n+1)));
          phiX = (gp(n+1,m)*A(n,m) - gp(n-1,m)*B(n,m) + imag*m*gt(n,m)) * (1/(Real)(std::max<Long>(n,1)*(n+1)));
        }

        auto phiV = (phiG * (n + 0) - phiY) * (1/(Real)(2*n + 1));
        auto phiW = (phiG * (n + 1) + phiY) * (1/(Real)(2*n + 1));

        if (n==0) {
          phiW = 0;
          phiX = 0;
        }
        write_coeff(phiV, B2, i+0, p0, n, m);
        write_coeff(phiW, B2, i+1, p0, n, m);
        write_coeff(phiX, B2, i+2, p0, n, m);
      }
    }
  }

  SHCArrange0(B2, p0, S, arrange);
}

template <class Real> void SphericalHarmonics<Real>::VecSHC2Grid(const Vector<Real>& S, SHCArrange arrange, Long p0, Long Nt, Long Np, Vector<Real>& X) {
  Vector<Real> B0;
  SHCArrange1(S, arrange, p0, B0);

  Long p_ = p0 + 1;
  Long M0 = (p0+1)*(p0+1);
  Long M_ = (p_+1)*(p_+1);
  Long N = B0.Dim() / M0;
  assert(B0.Dim() == N*M0);
  assert(N % COORD_DIM == 0);

  Vector<Real> B1(N*M_);
  const Complex<Real> imag(0,1);
  for (Long i=0; i<N; i+=COORD_DIM) {
    for (Long m=0; m<=p_; m++) {
      for (Long n=m; n<=p_; n++) {
        auto read_coeff = [&](const Vector<Real>& coeff, Long i, Long p, Long n, Long m) {
          Complex<Real> c;
          if (0<=m && m<=n && n<=p) {
            Long idx_real = ((2*p-m+3)*m - (m?p+1:0))*N + (p+1-m)*i - m + n;
            Long idx_imag = idx_real + (p+1-m)*N;
            c.real = coeff[idx_real];
            if (m) c.imag = coeff[idx_imag];
          }
          return c;
        };
        auto write_coeff = [&](Complex<Real> c, Vector<Real>& coeff, Long i, Long p, Long n, Long m) {
          if (0<=m && m<=n && n<=p) {
            Long idx_real = ((2*p-m+3)*m - (m?p+1:0))*N + (p+1-m)*i - m + n;
            Long idx_imag = idx_real + (p+1-m)*N;
            coeff[idx_real] = c.real;
            if (m) coeff[idx_imag] = c.imag;
          }
        };

        auto phiG = [&](Long n, Long m) {
          auto phiV = read_coeff(B0, i+0, p0, n, m);
          auto phiW = read_coeff(B0, i+1, p0, n, m);
          return phiV + phiW;
        };
        auto phiY = [&](Long n, Long m) {
          auto phiV = read_coeff(B0, i+0, p0, n, m);
          auto phiW = read_coeff(B0, i+1, p0, n, m);
          return phiW * n - phiV * (n + 1);
        };
        auto phiX = [&](Long n, Long m) {
          return read_coeff(B0, i+2, p0, n, m);
        };

        Complex<Real> gr, gt, gp;
        { // (gt, gp) <-- (phiG, phiX)
          auto A = [&](Long n, Long m) { return (0<=n && m<=n && n<=p_ ? sqrt<Real>(n*n * ((n+1)*(n+1) - m*m) / (Real)((2*n+1)*(2*n+3))) : 0); };
          auto B = [&](Long n, Long m) { return (0<=n && m<=n && n<=p_ ? sqrt<Real>((n+1)*(n+1) * (n*n - m*m) / (Real)((2*n+1)*(2*n-1))) : 0); };
          gr = phiY(n,m);
          gt = phiG(n-1,m)*A(n-1,m) - phiG(n+1,m)*B(n+1,m) - imag*m*phiX(n,m);
          gp = phiX(n-1,m)*A(n-1,m) - phiX(n+1,m)*B(n+1,m) + imag*m*phiG(n,m);
        }

        write_coeff(gr, B1, i+0, p_, n, m);
        write_coeff(gt, B1, i+1, p_, n, m);
        write_coeff(gp, B1, i+2, p_, n, m);
      }
    }
  }

  { // Set X
    SHC2Grid_(B1, p_, Nt, Np, &X);

    Vector<Real> sin_phi(Np), cos_phi(Np);
    for (Long i = 0; i < Np; i++) {
      sin_phi[i] = sin(2 * const_pi<Real>() * i / Np);
      cos_phi[i] = cos(2 * const_pi<Real>() * i / Np);
    }
    const auto& Y = LegendreNodes(Nt - 1);
    assert(Y.Dim() == Nt);
    Long Ngrid = Nt * Np;
    for (Long k = 0; k < N; k+=COORD_DIM) {
      for (Long i = 0; i < Nt; i++) {
        Real sin_theta = sqrt<Real>(1 - Y[i]*Y[i]);
        Real cos_theta = Y[i];
        Real s = 1 / sin_theta;
        auto X_ = X.begin() + (k*Nt+i)*Np;
        for (Long j = 0; j < Np; j++) {
          StaticArray<Real,3> in;
          in[0] = X_[0*Ngrid+j];
          in[1] = X_[1*Ngrid+j] * s;
          in[2] = X_[2*Ngrid+j] * s;

          StaticArray<Real,9> M;
          M[0] = sin_theta*cos_phi[j]; M[1] = sin_theta*sin_phi[j]; M[2] = cos_theta;
          M[3] = cos_theta*cos_phi[j]; M[4] = cos_theta*sin_phi[j]; M[5] =-sin_theta;
          M[6] =          -sin_phi[j]; M[7] =           cos_phi[j]; M[8] =         0;

          X_[0*Ngrid+j] = ( M[0] * in[0] + M[3] * in[1] + M[6] * in[2] );
          X_[1*Ngrid+j] = ( M[1] * in[0] + M[4] * in[1] + M[7] * in[2] );
          X_[2*Ngrid+j] = ( M[2] * in[0] + M[5] * in[1] + M[8] * in[2] );
        }
      }
    }
  }
}

template <class Real> void SphericalHarmonics<Real>::VecSHCEval(const Vector<Real>& S, SHCArrange arrange, Long p0, const Vector<Real>& cos_theta_phi, Vector<Real>& X) {
  Long M = (p0+1) * (p0+1);

  Long dof;
  Matrix<Real> B1;
  { // Set B1, dof
    Vector<Real> B0;
    SHCArrange1(S, arrange, p0, B0);
    dof = B0.Dim() / M / COORD_DIM;
    assert(B0.Dim() == dof * COORD_DIM * M);

    B1.ReInit(dof, COORD_DIM * M);
    Vector<Real> B1_(B1.Dim(0) * B1.Dim(1), B1.begin(), false);
    SHCArrange0(B0, p0, B1_, SHCArrange::COL_MAJOR_NONZERO);
  }
  assert(B1.Dim(1) == COORD_DIM * M);
  assert(B1.Dim(0) == dof);

  Matrix<Real> SHBasis;
  VecSHBasisEval(p0, cos_theta_phi, SHBasis);
  assert(SHBasis.Dim(1) == COORD_DIM * M);
  Long N = SHBasis.Dim(0) / COORD_DIM;

  { // Set X
    if (X.Dim() != N * dof * COORD_DIM) X.ReInit(N * dof * COORD_DIM);
    for (Long k0 = 0; k0 < N; k0++) {
      for (Long k1 = 0; k1 < dof; k1++) {
        StaticArray<Real,COORD_DIM> in;
        for (Long j = 0; j < COORD_DIM; j++) {
          in[j] = 0;
          for (Long i = 0; i < COORD_DIM * M; i++) {
            in[j] += B1[k1][i] * SHBasis[k0 * COORD_DIM + j][i];
          }
        }

        StaticArray<Real,9> M;
        Real cos_theta = cos_theta_phi[k0 * 2 + 0];
        Real sin_theta = sqrt<Real>(1 - cos_theta * cos_theta);
        Real cos_phi = cos(cos_theta_phi[k0 * 2 + 1]);
        Real sin_phi = sin(cos_theta_phi[k0 * 2 + 1]);
        M[0] = sin_theta*cos_phi; M[1] = sin_theta*sin_phi; M[2] = cos_theta;
        M[3] = cos_theta*cos_phi; M[4] = cos_theta*sin_phi; M[5] =-sin_theta;
        M[6] =          -sin_phi; M[7] =           cos_phi; M[8] =         0;

        X[(k0 * dof + k1) * COORD_DIM + 0] = M[0] * in[0] + M[3] * in[1] + M[6] * in[2];
        X[(k0 * dof + k1) * COORD_DIM + 1] = M[1] * in[0] + M[4] * in[1] + M[7] * in[2];
        X[(k0 * dof + k1) * COORD_DIM + 2] = M[2] * in[0] + M[5] * in[1] + M[8] * in[2];
      }
    }
  }
}

template <class Real> void SphericalHarmonics<Real>::StokesEvalSL(const Vector<Real>& S, SHCArrange arrange, Long p0, const Vector<Real>& coord, bool interior, Vector<Real>& X) {
  Long M = (p0+1) * (p0+1);

  Long dof;
  Matrix<Real> B1;
  { // Set B1, dof
    Vector<Real> B0;
    SHCArrange1(S, arrange, p0, B0);
    dof = B0.Dim() / M / COORD_DIM;
    assert(B0.Dim() == dof * COORD_DIM * M);

    B1.ReInit(dof, COORD_DIM * M);
    Vector<Real> B1_(B1.Dim(0) * B1.Dim(1), B1.begin(), false);
    SHCArrange0(B0, p0, B1_, SHCArrange::COL_MAJOR_NONZERO);
  }
  assert(B1.Dim(1) == COORD_DIM * M);
  assert(B1.Dim(0) == dof);

  Long N;
  Matrix<Real> SHBasis;
  Vector<Real> R, cos_theta_phi;
  { // Set N, R, SHBasis
    N = coord.Dim() / COORD_DIM;
    assert(coord.Dim() == N * COORD_DIM);

    R.ReInit(N);
    cos_theta_phi.ReInit(2 * N);
    for (Long i = 0; i < N; i++) { // Set R, cos_theta_phi
      ConstIterator<Real> x = coord.begin() + i * COORD_DIM;
      R[i] = sqrt<Real>(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);
      cos_theta_phi[i * 2 + 0] = x[2] / R[i];
      cos_theta_phi[i * 2 + 1] = atan2(x[1], x[0]); // TODO: works only for float and double
    }
    VecSHBasisEval(p0, cos_theta_phi, SHBasis);
    assert(SHBasis.Dim(1) == COORD_DIM * M);
    assert(SHBasis.Dim(0) == N * COORD_DIM);
  }

  Matrix<Real> StokesOp(SHBasis.Dim(0), SHBasis.Dim(1));
  for (Long i = 0; i < N; i++) { // Set StokesOp
    for (Long m = 0; m <= p0; m++) {
      for (Long n = m; n <= p0; n++) {
        auto read_coeff = [&](Long n, Long m, Long k0, Long k1) {
          Complex<Real> c;
          if (0 <= m && m <= n && n <= p0 && 0 <= k0 && k0 < COORD_DIM && 0 <= k1 && k1 < COORD_DIM) {
            Long idx = (2 * p0 - m + 2) * m - (m ? p0+1 : 0) + n;
            c.real = SHBasis[i * COORD_DIM + k1][k0 * M + idx];
            if (m) {
              idx += (p0+1-m);
              c.imag = SHBasis[i * COORD_DIM + k1][k0 * M + idx];
            }
          }
          return c;
        };
        auto write_coeff = [&](Complex<Real> c, Long n, Long m, Long k0, Long k1) {
          if (0 <= m && m <= n && n <= p0 && 0 <= k0 && k0 < COORD_DIM && 0 <= k1 && k1 < COORD_DIM) {
            Long idx = (2 * p0 - m + 2) * m - (m ? p0+1 : 0) + n;
            StokesOp[i * COORD_DIM + k1][k0 * M + idx] = c.real;
            if (m) {
              idx += (p0+1-m);
              StokesOp[i * COORD_DIM + k1][k0 * M + idx] = c.imag;
            }
          }
        };

        auto Vr = read_coeff(n, m, 0, 0);
        auto Vt = read_coeff(n, m, 0, 1);
        auto Vp = read_coeff(n, m, 0, 2);

        auto Wr = read_coeff(n, m, 1, 0);
        auto Wt = read_coeff(n, m, 1, 1);
        auto Wp = read_coeff(n, m, 1, 2);

        auto Xr = read_coeff(n, m, 2, 0);
        auto Xt = read_coeff(n, m, 2, 1);
        auto Xp = read_coeff(n, m, 2, 2);

        Complex<Real> SVr, SVt, SVp;
        Complex<Real> SWr, SWt, SWp;
        Complex<Real> SXr, SXt, SXp;

        if (interior) {
          Real a,b;
          a = n / (Real)((2*n+1) * (2*n+3)) * pow<Real>(R[i], n+1);
          b = -(n+1) / (Real)(4*n+2) * (pow<Real>(R[i], n-1) - pow<Real>(R[i], n+1));
          SVr = a * Vr + b * Wr;
          SVt = a * Vt + b * Wt;
          SVp = a * Vp + b * Wp;

          a = (n+1) / (Real)((2*n+1) * (2*n-1)) * pow<Real>(R[i], n-1);
          SWr = a * Wr;
          SWt = a * Wt;
          SWp = a * Wp;

          a = 1 / (Real)(2*n+1) * pow<Real>(R[i], n);
          SXr = a * Xr;
          SXt = a * Xt;
          SXp = a * Xp;
        } else {
          Real a,b;
          a = n / (Real)((2*n+1) * (2*n+3)) * pow<Real>(R[i], -n-2);
          SVr = a * Vr;
          SVt = a * Vt;
          SVp = a * Vp;

          a = (n+1) / (Real)((2*n+1) * (2*n-1)) * pow<Real>(R[i], -n);
          b = n / (Real)(4*n+2) * (pow<Real>(R[i], -n-2) - pow<Real>(R[i], -n));
          SWr = a * Wr + b * Vr;
          SWt = a * Wt + b * Vt;
          SWp = a * Wp + b * Vp;

          a = 1 / (Real)(2*n+1) * pow<Real>(R[i], -n-1);
          SXr = a * Xr;
          SXt = a * Xt;
          SXp = a * Xp;
        }

        write_coeff(SVr, n, m, 0, 0);
        write_coeff(SVt, n, m, 0, 1);
        write_coeff(SVp, n, m, 0, 2);

        write_coeff(SWr, n, m, 1, 0);
        write_coeff(SWt, n, m, 1, 1);
        write_coeff(SWp, n, m, 1, 2);

        write_coeff(SXr, n, m, 2, 0);
        write_coeff(SXt, n, m, 2, 1);
        write_coeff(SXp, n, m, 2, 2);
      }
    }
  }

  { // Set X
    if (X.Dim() != N * dof * COORD_DIM) X.ReInit(N * dof * COORD_DIM);
    for (Long k0 = 0; k0 < N; k0++) {
      for (Long k1 = 0; k1 < dof; k1++) {
        StaticArray<Real,COORD_DIM> in;
        for (Long j = 0; j < COORD_DIM; j++) {
          in[j] = 0;
          for (Long i = 0; i < COORD_DIM * M; i++) {
            in[j] += B1[k1][i] * StokesOp[k0 * COORD_DIM + j][i];
          }
        }

        StaticArray<Real,9> M;
        Real cos_theta = cos_theta_phi[k0 * 2 + 0];
        Real sin_theta = sqrt<Real>(1 - cos_theta * cos_theta);
        Real cos_phi = cos(cos_theta_phi[k0 * 2 + 1]);
        Real sin_phi = sin(cos_theta_phi[k0 * 2 + 1]);
        M[0] = sin_theta*cos_phi; M[1] = sin_theta*sin_phi; M[2] = cos_theta;
        M[3] = cos_theta*cos_phi; M[4] = cos_theta*sin_phi; M[5] =-sin_theta;
        M[6] =          -sin_phi; M[7] =           cos_phi; M[8] =         0;

        X[(k0 * dof + k1) * COORD_DIM + 0] = M[0] * in[0] + M[3] * in[1] + M[6] * in[2];
        X[(k0 * dof + k1) * COORD_DIM + 1] = M[1] * in[0] + M[4] * in[1] + M[7] * in[2];
        X[(k0 * dof + k1) * COORD_DIM + 2] = M[2] * in[0] + M[5] * in[1] + M[8] * in[2];
      }
    }
  }
}

template <class Real> void SphericalHarmonics<Real>::StokesEvalDL(const Vector<Real>& S, SHCArrange arrange, Long p0, const Vector<Real>& coord, bool interior, Vector<Real>& X) {
  Long M = (p0+1) * (p0+1);

  Long dof;
  Matrix<Real> B1;
  { // Set B1, dof
    Vector<Real> B0;
    SHCArrange1(S, arrange, p0, B0);
    dof = B0.Dim() / M / COORD_DIM;
    assert(B0.Dim() == dof * COORD_DIM * M);

    B1.ReInit(dof, COORD_DIM * M);
    Vector<Real> B1_(B1.Dim(0) * B1.Dim(1), B1.begin(), false);
    SHCArrange0(B0, p0, B1_, SHCArrange::COL_MAJOR_NONZERO);
  }
  assert(B1.Dim(1) == COORD_DIM * M);
  assert(B1.Dim(0) == dof);

  Long N;
  Matrix<Real> SHBasis;
  Vector<Real> R, cos_theta_phi;
  { // Set N, R, SHBasis
    N = coord.Dim() / COORD_DIM;
    assert(coord.Dim() == N * COORD_DIM);

    R.ReInit(N);
    cos_theta_phi.ReInit(2 * N);
    for (Long i = 0; i < N; i++) { // Set R, cos_theta_phi
      ConstIterator<Real> x = coord.begin() + i * COORD_DIM;
      R[i] = sqrt<Real>(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);
      cos_theta_phi[i * 2 + 0] = x[2] / R[i];
      cos_theta_phi[i * 2 + 1] = atan2(x[1], x[0]); // TODO: works only for float and double
    }
    VecSHBasisEval(p0, cos_theta_phi, SHBasis);
    assert(SHBasis.Dim(1) == COORD_DIM * M);
    assert(SHBasis.Dim(0) == N * COORD_DIM);
  }

  Matrix<Real> StokesOp(SHBasis.Dim(0), SHBasis.Dim(1));
  for (Long i = 0; i < N; i++) { // Set StokesOp
    for (Long m = 0; m <= p0; m++) {
      for (Long n = m; n <= p0; n++) {
        auto read_coeff = [&](Long n, Long m, Long k0, Long k1) {
          Complex<Real> c;
          if (0 <= m && m <= n && n <= p0 && 0 <= k0 && k0 < COORD_DIM && 0 <= k1 && k1 < COORD_DIM) {
            Long idx = (2 * p0 - m + 2) * m - (m ? p0+1 : 0) + n;
            c.real = SHBasis[i * COORD_DIM + k1][k0 * M + idx];
            if (m) {
              idx += (p0+1-m);
              c.imag = SHBasis[i * COORD_DIM + k1][k0 * M + idx];
            }
          }
          return c;
        };
        auto write_coeff = [&](Complex<Real> c, Long n, Long m, Long k0, Long k1) {
          if (0 <= m && m <= n && n <= p0 && 0 <= k0 && k0 < COORD_DIM && 0 <= k1 && k1 < COORD_DIM) {
            Long idx = (2 * p0 - m + 2) * m - (m ? p0+1 : 0) + n;
            StokesOp[i * COORD_DIM + k1][k0 * M + idx] = c.real;
            if (m) {
              idx += (p0+1-m);
              StokesOp[i * COORD_DIM + k1][k0 * M + idx] = c.imag;
            }
          }
        };

        auto Vr = read_coeff(n, m, 0, 0);
        auto Vt = read_coeff(n, m, 0, 1);
        auto Vp = read_coeff(n, m, 0, 2);

        auto Wr = read_coeff(n, m, 1, 0);
        auto Wt = read_coeff(n, m, 1, 1);
        auto Wp = read_coeff(n, m, 1, 2);

        auto Xr = read_coeff(n, m, 2, 0);
        auto Xt = read_coeff(n, m, 2, 1);
        auto Xp = read_coeff(n, m, 2, 2);

        Complex<Real> SVr, SVt, SVp;
        Complex<Real> SWr, SWt, SWp;
        Complex<Real> SXr, SXt, SXp;

        if (interior) {
          Real a,b;
          a = -2*n*(n+2) / (Real)((2*n+1) * (2*n+3)) * pow<Real>(R[i], n+1);
          b = -(n+1)*(n+2) / (Real)(2*n+1) * (pow<Real>(R[i], n+1) - pow<Real>(R[i], n-1));
          SVr = a * Vr + b * Wr;
          SVt = a * Vt + b * Wt;
          SVp = a * Vp + b * Wp;

          a = -(2*n*n+1) / (Real)((2*n+1) * (2*n-1)) * pow<Real>(R[i], n-1);
          SWr = a * Wr;
          SWt = a * Wt;
          SWp = a * Wp;

          a = -(n+2) / (Real)(2*n+1) * pow<Real>(R[i], n);
          SXr = a * Xr;
          SXt = a * Xt;
          SXp = a * Xp;
        } else {
          Real a,b;
          a = (2*n*n+4*n+3) / (Real)((2*n+1) * (2*n+3)) * pow<Real>(R[i], -n-2);
          SVr = a * Vr;
          SVt = a * Vt;
          SVp = a * Vp;

          a = 2*(n+1)*(n-1) / (Real)((2*n+1) * (2*n-1)) * pow<Real>(R[i], -n);
          b = 2*n*(n-1) / (Real)(4*n+2) * (pow<Real>(R[i], -n-2) - pow<Real>(R[i], -n));
          SWr = a * Wr + b * Vr;
          SWt = a * Wt + b * Vt;
          SWp = a * Wp + b * Vp;

          a = (n-1) / (Real)(2*n+1) * pow<Real>(R[i], -n-1);
          SXr = a * Xr;
          SXt = a * Xt;
          SXp = a * Xp;
        }

        write_coeff(SVr, n, m, 0, 0);
        write_coeff(SVt, n, m, 0, 1);
        write_coeff(SVp, n, m, 0, 2);

        write_coeff(SWr, n, m, 1, 0);
        write_coeff(SWt, n, m, 1, 1);
        write_coeff(SWp, n, m, 1, 2);

        write_coeff(SXr, n, m, 2, 0);
        write_coeff(SXt, n, m, 2, 1);
        write_coeff(SXp, n, m, 2, 2);
      }
    }
  }

  { // Set X
    if (X.Dim() != N * dof * COORD_DIM) X.ReInit(N * dof * COORD_DIM);
    for (Long k0 = 0; k0 < N; k0++) {
      for (Long k1 = 0; k1 < dof; k1++) {
        StaticArray<Real,COORD_DIM> in;
        for (Long j = 0; j < COORD_DIM; j++) {
          in[j] = 0;
          for (Long i = 0; i < COORD_DIM * M; i++) {
            in[j] += B1[k1][i] * StokesOp[k0 * COORD_DIM + j][i];
          }
        }

        StaticArray<Real,9> M;
        Real cos_theta = cos_theta_phi[k0 * 2 + 0];
        Real sin_theta = sqrt<Real>(1 - cos_theta * cos_theta);
        Real cos_phi = cos(cos_theta_phi[k0 * 2 + 1]);
        Real sin_phi = sin(cos_theta_phi[k0 * 2 + 1]);
        M[0] = sin_theta*cos_phi; M[1] = sin_theta*sin_phi; M[2] = cos_theta;
        M[3] = cos_theta*cos_phi; M[4] = cos_theta*sin_phi; M[5] =-sin_theta;
        M[6] =          -sin_phi; M[7] =           cos_phi; M[8] =         0;

        X[(k0 * dof + k1) * COORD_DIM + 0] = M[0] * in[0] + M[3] * in[1] + M[6] * in[2];
        X[(k0 * dof + k1) * COORD_DIM + 1] = M[1] * in[0] + M[4] * in[1] + M[7] * in[2];
        X[(k0 * dof + k1) * COORD_DIM + 2] = M[2] * in[0] + M[5] * in[1] + M[8] * in[2];
      }
    }
  }
}






template <class Real> void SphericalHarmonics<Real>::Grid2SHC_(const Vector<Real>& X, Long Nt, Long Np, Long p1, Vector<Real>& B1){
  const auto& Mf = OpFourierInv(Np);
  assert(Mf.Dim(0) == Np);

  const std::vector<Matrix<Real>>& Ml = SphericalHarmonics<Real>::MatLegendreInv(Nt-1,p1);
  assert((Long)Ml.size() == p1+1);

  Long N = X.Dim() / (Np*Nt);
  assert(X.Dim() == N*Np*Nt);

  Vector<Real> B0((2*p1+1) * N*Nt);
  #pragma omp parallel
  { // B0 <-- Transpose(FFT(X))
    Integer tid=omp_get_thread_num();
    Integer omp_p=omp_get_num_threads();
    Long a=(tid+0)*N*Nt/omp_p;
    Long b=(tid+1)*N*Nt/omp_p;

    Vector<Real> buff(Mf.Dim(1));
    Long fft_coeff_len = std::min(buff.Dim(), 2*p1+2);
    Matrix<Real> B0_(2*p1+1, N*Nt, B0.begin(), false);
    const Matrix<Real> MX(N * Nt, Np, (Iterator<Real>)X.begin(), false);
    for (Long i = a; i < b; i++) {
      { // buff <-- FFT(Xi)
        const Vector<Real> Xi(Np, (Iterator<Real>)X.begin() + Np * i, false);
        Mf.Execute(Xi, buff);
      }
      { // B0 <-- Transpose(buff)
        B0_[0][i] = buff[0]; // skipping buff[1] == 0
        for (Long j = 2; j < fft_coeff_len; j++) B0_[j-1][i] = buff[j];
        for (Long j = fft_coeff_len; j < 2*p1+2; j++) B0_[j-1][i] = 0;
      }
    }
  }

  if (B1.Dim() != N*(p1+1)*(p1+1)) B1.ReInit(N*(p1+1)*(p1+1));
  #pragma omp parallel
  { // Evaluate Legendre polynomial
    Integer tid=omp_get_thread_num();
    Integer omp_p=omp_get_num_threads();

    Long offset0=0;
    Long offset1=0;
    for (Long i = 0; i < p1+1; i++) {
      Long N_ = (i==0 ? N : 2*N);
      Matrix<Real> Min (N_, Nt    , B0.begin()+offset0, false);
      Matrix<Real> Mout(N_, p1+1-i, B1.begin()+offset1, false);
      { // Mout = Min * Ml[i]  // split between threads
        Long a=(tid+0)*N_/omp_p;
        Long b=(tid+1)*N_/omp_p;
        if (a < b) {
          Matrix<Real> Min_ (b-a, Min .Dim(1), Min [a], false);
          Matrix<Real> Mout_(b-a, Mout.Dim(1), Mout[a], false);
          Matrix<Real>::GEMM(Mout_,Min_,Ml[i]);
        }
      }
      offset0+=Min .Dim(0)*Min .Dim(1);
      offset1+=Mout.Dim(0)*Mout.Dim(1);
    }
    assert(offset0 == B0.Dim());
    assert(offset1 == B1.Dim());
  }
  B1 *= 1 / sqrt<Real>(4 * const_pi<Real>() * Np); // Scaling to match Zydrunas Fortran code.
}
template <class Real> void SphericalHarmonics<Real>::SHCArrange0(const Vector<Real>& B1, Long p1, Vector<Real>& S, SHCArrange arrange){
  Long M = (p1+1)*(p1+1);
  Long N = B1.Dim() / M;
  assert(B1.Dim() == N*M);
  if (arrange == SHCArrange::ALL) { // S <-- Rearrange(B1)
    Long M = 2*(p1+1)*(p1+1);
    if(S.Dim() != N * M) S.ReInit(N * M);
    #pragma omp parallel
    { // S <-- Rearrange(B1)
      Integer tid=omp_get_thread_num();
      Integer omp_p=omp_get_num_threads();

      Long a=(tid+0)*N/omp_p;
      Long b=(tid+1)*N/omp_p;
      for (Long i = a; i < b; i++) {
        Long offset = 0;
        for (Long j = 0; j < p1+1; j++) {
          Long len = p1+1 - j;
          if (1) { // Set Real(S_n^m) for m=j and n=j..p
            ConstIterator<Real> B_ = B1.begin() + i*len + N*offset;
            Iterator<Real>      S_ = S .begin() + i*M   + j*(p1+1)*2 + j*2 + 0;
            for (Long k = 0; k < len; k++) S_[k * (p1+1)*2] = B_[k];
            offset += len;
          }
          if (j) { // Set Imag(S_n^m) for m=j and n=j..p
            ConstIterator<Real> B_ = B1.begin() + i*len + N*offset;
            Iterator<Real>      S_ = S .begin() + i*M   + j*(p1+1)*2 + j*2 + 1;
            for (Long k = 0; k < len; k++) S_[k * (p1+1)*2] = B_[k];
            offset += len;
          } else {
            Iterator<Real>      S_ = S .begin() + i*M   + j*(p1+1)*2 + j*2 + 1;
            for (Long k = 0; k < len; k++) S_[k * (p1+1)*2] = 0;
          }
        }
      }
    }
  }
  if (arrange == SHCArrange::ROW_MAJOR) { // S <-- Rearrange(B1)
    Long M = (p1+1)*(p1+2);
    if(S.Dim() != N * M) S.ReInit(N * M);
    #pragma omp parallel
    { // S <-- Rearrange(B1)
      Integer tid=omp_get_thread_num();
      Integer omp_p=omp_get_num_threads();

      Long a=(tid+0)*N/omp_p;
      Long b=(tid+1)*N/omp_p;
      for (Long i = a; i < b; i++) {
        Long offset = 0;
        for (Long j = 0; j < p1+1; j++) {
          Long len = p1+1 - j;
          if (1) { // Set Real(S_n^m) for m=j and n=j..p
            ConstIterator<Real> B_ = B1.begin() + i*len + N*offset;
            Iterator<Real>      S_ = S .begin() + i*M + 0;
            for (Long k=0;k<len;k++) S_[(j+k)*(j+k+1) + 2*j] = B_[k];
            offset += len;
          }
          if (j) { // Set Imag(S_n^m) for m=j and n=j..p
            ConstIterator<Real> B_ = B1.begin() + i*len + N*offset;
            Iterator<Real>      S_ = S .begin() + i*M + 1;
            for (Long k=0;k<len;k++) S_[(j+k)*(j+k+1) + 2*j] = B_[k];
            offset += len;
          } else {
            Iterator<Real> S_ = S .begin() + i*M + 1;
            for (Long k=0;k<len;k++) S_[(j+k)*(j+k+1) + 2*j] = 0;
          }
        }
      }
    }
  }
  if (arrange == SHCArrange::COL_MAJOR_NONZERO) { // S <-- Rearrange(B1)
    Long M = (p1+1)*(p1+1);
    if(S.Dim() != N * M) S.ReInit(N * M);
    #pragma omp parallel
    { // S <-- Rearrange(B1)
      Integer tid=omp_get_thread_num();
      Integer omp_p=omp_get_num_threads();

      Long a=(tid+0)*N/omp_p;
      Long b=(tid+1)*N/omp_p;
      for (Long i = a; i < b; i++) {
        Long offset = 0;
        for (Long j = 0; j <  p1+1; j++) {
          Long len = p1+1 - j;
          if (1) { // Set Real(S_n^m) for m=j and n=j..p
            ConstIterator<Real> B_ = B1.begin() + i*len + N*offset;
            Iterator<Real>      S_ = S .begin() + i*M   + offset;
            for (Long k = 0; k < len; k++) S_[k] = B_[k];
            offset += len;
          }
          if (j) { // Set Imag(S_n^m) for m=j and n=j..p
            ConstIterator<Real> B_ = B1.begin() + i*len + N*offset;
            Iterator<Real>      S_ = S .begin() + i*M   + offset;
            for (Long k = 0; k < len; k++) S_[k] = B_[k];
            offset += len;
          }
        }
      }
    }
  }
}

template <class Real> void SphericalHarmonics<Real>::SHCArrange1(const Vector<Real>& S, SHCArrange arrange, Long p0, Vector<Real>& B0){
  Long M, N;
  { // Set M, N
    M = 0;
    if (arrange == SHCArrange::ALL) M = 2*(p0+1)*(p0+1);
    if (arrange == SHCArrange::ROW_MAJOR) M = (p0+1)*(p0+2);
    if (arrange == SHCArrange::COL_MAJOR_NONZERO) M = (p0+1)*(p0+1);
    if (M == 0) return;
    N = S.Dim() / M;
    assert(S.Dim() == N * M);
  }

  if (B0.Dim() != N*(p0+1)*(p0+1)) B0.ReInit(N*(p0+1)*(p0+1));
  if (arrange == SHCArrange::ALL) { // B0 <-- Rearrange(S)
    #pragma omp parallel
    { // B0 <-- Rearrange(S)
      Integer tid=omp_get_thread_num();
      Integer omp_p=omp_get_num_threads();

      Long a=(tid+0)*N/omp_p;
      Long b=(tid+1)*N/omp_p;
      for (Long i = a; i < b; i++) {
        Long offset = 0;
        for (Long j = 0; j < p0+1; j++) {
          Long len = p0+1 - j;
          if (1) { // Get Real(S_n^m) for m=j and n=j..p
            Iterator<Real>      B_ = B0.begin() + i*len + N*offset;
            ConstIterator<Real> S_ = S .begin() + i*M   + j*(p0+1)*2 + j*2 + 0;
            for (Long k = 0; k < len; k++) B_[k] = S_[k * (p0+1)*2];
            offset += len;
          }
          if (j) { // Get Imag(S_n^m) for m=j and n=j..p
            Iterator<Real>      B_ = B0.begin() + i*len + N*offset;
            ConstIterator<Real> S_ = S .begin() + i*M   + j*(p0+1)*2 + j*2 + 1;
            for (Long k = 0; k < len; k++) B_[k] = S_[k * (p0+1)*2];
            offset += len;
          }
        }
      }
    }
  }
  if (arrange == SHCArrange::ROW_MAJOR) { // B0 <-- Rearrange(S)
    #pragma omp parallel
    { // B0 <-- Rearrange(S)
      Integer tid=omp_get_thread_num();
      Integer omp_p=omp_get_num_threads();

      Long a=(tid+0)*N/omp_p;
      Long b=(tid+1)*N/omp_p;
      for (Long i = a; i < b; i++) {
        Long offset = 0;
        for (Long j = 0; j < p0+1; j++) {
          Long len = p0+1 - j;
          if (1) { // Get Real(S_n^m) for m=j and n=j..p
            Iterator<Real>      B_ = B0.begin() + i*len + N*offset;
            ConstIterator<Real> S_ = S .begin() + i*M + 0;
            for (Long k=0;k<len;k++) B_[k] = S_[(j+k)*(j+k+1) + 2*j];
            offset += len;
          }
          if (j) { // Get Imag(S_n^m) for m=j and n=j..p
            Iterator<Real>      B_ = B0.begin() + i*len + N*offset;
            ConstIterator<Real> S_ = S .begin() + i*M + 1;
            for (Long k=0;k<len;k++) B_[k] = S_[(j+k)*(j+k+1) + 2*j];
            offset += len;
          }
        }
      }
    }
  }
  if (arrange == SHCArrange::COL_MAJOR_NONZERO) { // B0 <-- Rearrange(S)
    #pragma omp parallel
    { // B0 <-- Rearrange(S)
      Integer tid=omp_get_thread_num();
      Integer omp_p=omp_get_num_threads();

      Long a=(tid+0)*N/omp_p;
      Long b=(tid+1)*N/omp_p;
      for (Long i = a; i < b; i++) {
        Long offset = 0;
        for (Long j = 0; j <  p0+1; j++) {
          Long len = p0+1 - j;
          if (1) { // Get Real(S_n^m) for m=j and n=j..p
            Iterator<Real>      B_ = B0.begin() + i*len + N*offset;
            ConstIterator<Real> S_ = S .begin() + i*M   + offset;
            for (Long k = 0; k < len; k++) B_[k] = S_[k];
            offset += len;
          }
          if (j) { // Get Imag(S_n^m) for m=j and n=j..p
            Iterator<Real>      B_ = B0.begin() + i*len + N*offset;
            ConstIterator<Real> S_ = S .begin() + i*M   + offset;
            for (Long k = 0; k < len; k++) B_[k] = S_[k];
            offset += len;
          }
        }
      }
    }
  }
}
template <class Real> void SphericalHarmonics<Real>::SHC2Grid_(const Vector<Real>& B0, Long p0, Long Nt, Long Np, Vector<Real>* X, Vector<Real>* X_phi, Vector<Real>* X_theta){
  const auto& Mf = OpFourier(Np);
  assert(Mf.Dim(1) == Np);

  const std::vector<Matrix<Real>>& Ml =SphericalHarmonics<Real>::MatLegendre    (p0,Nt-1);
  const std::vector<Matrix<Real>>& Mdl=SphericalHarmonics<Real>::MatLegendreGrad(p0,Nt-1);
  assert((Long)Ml .size() == p0+1);
  assert((Long)Mdl.size() == p0+1);

  Long N = B0.Dim() / ((p0+1)*(p0+1));
  assert(B0.Dim() == N*(p0+1)*(p0+1));

  if(X       && X      ->Dim()!=N*Np*Nt) X      ->ReInit(N*Np*Nt);
  if(X_theta && X_theta->Dim()!=N*Np*Nt) X_theta->ReInit(N*Np*Nt);
  if(X_phi   && X_phi  ->Dim()!=N*Np*Nt) X_phi  ->ReInit(N*Np*Nt);

  Vector<Real> B1(N*(2*p0+1)*Nt);
  if(X || X_phi){
    #pragma omp parallel
    { // Evaluate Legendre polynomial
      Integer tid=omp_get_thread_num();
      Integer omp_p=omp_get_num_threads();

      Long offset0=0;
      Long offset1=0;
      for(Long i=0;i<p0+1;i++){
        Long N_ = (i==0 ? N : 2*N);
        const Matrix<Real> Min (N_, p0+1-i, (Iterator<Real>)B0.begin()+offset0, false);
        Matrix<Real> Mout(N_, Nt    , B1.begin()+offset1, false);
        { // Mout = Min * Ml[i]  // split between threads
          Long a=(tid+0)*N_/omp_p;
          Long b=(tid+1)*N_/omp_p;
          if(a<b){
            const Matrix<Real> Min_ (b-a, Min .Dim(1), (Iterator<Real>)Min [a], false);
            Matrix<Real> Mout_(b-a, Mout.Dim(1), Mout[a], false);
            Matrix<Real>::GEMM(Mout_,Min_,Ml[i]);
          }
        }
        offset0+=Min .Dim(0)*Min .Dim(1);
        offset1+=Mout.Dim(0)*Mout.Dim(1);
      }
    }
    B1 *= sqrt<Real>(4 * const_pi<Real>() * Np); // Scaling to match Zydrunas Fortran code.

    #pragma omp parallel
    { // Transpose and evaluate Fourier
      Integer tid=omp_get_thread_num();
      Integer omp_p=omp_get_num_threads();

      Long a=(tid+0)*N*Nt/omp_p;
      Long b=(tid+1)*N*Nt/omp_p;

      Vector<Real> buff(Mf.Dim(0)); buff = 0;
      Long fft_coeff_len = std::min(buff.Dim(), 2*p0+2);
      Matrix<Real> B1_(2*p0+1, N*Nt, B1.begin(), false);
      for (Long i = a; i < b; i++) {
        { // buff <-- Transpose(B1)
          buff[0] = B1_[0][i];
          buff[1] = 0;
          for (Long j = 2; j < fft_coeff_len; j++) buff[j] = B1_[j-1][i];
          for (Long j = fft_coeff_len; j < buff.Dim(); j++) buff[j] = 0;
        }
        { // X <-- FFT(buff)
          Vector<Real> Xi(Np, X->begin() + Np * i, false);
          Mf.Execute(buff, Xi);
        }

        if(X_phi){ // Evaluate Fourier gradient
          { // buff <-- Transpose(B1)
            buff[0] = 0;
            buff[1] = 0;
            for (Long j = 2; j < fft_coeff_len; j++) buff[j] = B1_[j-1][i];
            for (Long j = fft_coeff_len; j < buff.Dim(); j++) buff[j] = 0;
            for (Long j = 1; j < buff.Dim()/2; j++) {
              Real x = buff[2*j+0];
              Real y = buff[2*j+1];
              buff[2*j+0] = -j*y;
              buff[2*j+1] =  j*x;
            }
          }
          { // X_phi <-- FFT(buff)
            Vector<Real> Xi(Np, X_phi->begin() + Np * i, false);
            Mf.Execute(buff, Xi);
          }
        }
      }
    }
  }
  if(X_theta){
    #pragma omp parallel
    { // Evaluate Legendre gradient
      Integer tid=omp_get_thread_num();
      Integer omp_p=omp_get_num_threads();

      Long offset0=0;
      Long offset1=0;
      for(Long i=0;i<p0+1;i++){
        Long N_ = (i==0 ? N : 2*N);
        const Matrix<Real> Min (N_, p0+1-i, (Iterator<Real>)B0.begin()+offset0, false);
        Matrix<Real> Mout(N_, Nt    , B1.begin()+offset1, false);
        { // Mout = Min * Mdl[i]  // split between threads
          Long a=(tid+0)*N_/omp_p;
          Long b=(tid+1)*N_/omp_p;
          if(a<b){
            const Matrix<Real> Min_ (b-a, Min .Dim(1), (Iterator<Real>)Min [a], false);
            Matrix<Real> Mout_(b-a, Mout.Dim(1), Mout[a], false);
            Matrix<Real>::GEMM(Mout_,Min_,Mdl[i]);
          }
        }
        offset0+=Min .Dim(0)*Min .Dim(1);
        offset1+=Mout.Dim(0)*Mout.Dim(1);
      }
    }
    B1 *= sqrt<Real>(4 * const_pi<Real>() * Np); // Scaling to match Zydrunas Fortran code.

    #pragma omp parallel
    { // Transpose and evaluate Fourier
      Integer tid=omp_get_thread_num();
      Integer omp_p=omp_get_num_threads();

      Long a=(tid+0)*N*Nt/omp_p;
      Long b=(tid+1)*N*Nt/omp_p;

      Vector<Real> buff(Mf.Dim(0)); buff = 0;
      Long fft_coeff_len = std::min(buff.Dim(), 2*p0+2);
      Matrix<Real> B1_(2*p0+1, N*Nt, B1.begin(), false);
      for (Long i = a; i < b; i++) {
        { // buff <-- Transpose(B1)
          buff[0] = B1_[0][i];
          buff[1] = 0;
          for (Long j = 2; j < fft_coeff_len; j++) buff[j] = B1_[j-1][i];
          for (Long j = fft_coeff_len; j < buff.Dim(); j++) buff[j] = 0;
        }
        { // Xi <-- FFT(buff)
          Vector<Real> Xi(Np, X_theta->begin() + Np * i, false);
          Mf.Execute(buff, Xi);
        }
      }
    }
  }
}


template <class Real> void SphericalHarmonics<Real>::LegPoly(Vector<Real>& poly_val, const Vector<Real>& X, Long degree){
  Long N = X.Dim();
  Long Npoly = (degree + 1) * (degree + 2) / 2;
  if (poly_val.Dim() != Npoly * N) poly_val.ReInit(Npoly * N);

  Real fact = 1 / sqrt<Real>(4 * const_pi<Real>());
  Vector<Real> u(N);
  for (Long n = 0; n < N; n++) {
    u[n] = (X[n]*X[n]<1 ? sqrt<Real>(1-X[n]*X[n]) : 0);
    poly_val[n] = fact;
  }

  Long idx = 0;
  Long idx_nxt = 0;
  for (Long i = 1; i <= degree; i++) {
    idx_nxt += N*(degree-i+2);
    Real c = sqrt<Real>((2*i+1)/(Real)(2*i));
    for (Long n = 0; n < N; n++) {
      poly_val[idx_nxt+n] = -poly_val[idx+n] * u[n] * c;
    }
    idx = idx_nxt;
  }

  idx = 0;
  for (Long m = 0; m < degree; m++) {
    for (Long n = 0; n < N; n++) {
      Real pmm = 0;
      Real pmmp1 = poly_val[idx+n];
      for (Long ll = m + 1; ll <= degree; ll++) {
        Real a = sqrt<Real>(((2*ll-1)*(2*ll+1)         ) / (Real)((ll-m)*(ll+m)         ));
        Real b = sqrt<Real>(((2*ll+1)*(ll+m-1)*(ll-m-1)) / (Real)((ll-m)*(ll+m)*(2*ll-3)));
        Real pll = X[n]*a*pmmp1 - b*pmm;
        pmm = pmmp1;
        pmmp1 = pll;
        poly_val[idx + N*(ll-m) + n] = pll;
      }
    }
    idx += N * (degree - m + 1);
  }
}

template <class Real> void SphericalHarmonics<Real>::LegPolyDeriv(Vector<Real>& poly_val, const Vector<Real>& X, Long degree){
  Long N = X.Dim();
  Long Npoly = (degree + 1) * (degree + 2) / 2;
  if (poly_val.Dim() != N * Npoly) poly_val.ReInit(N * Npoly);

  Vector<Real> leg_poly(Npoly * N);
  LegPoly(leg_poly, X, degree);

  for (Long m = 0; m <= degree; m++) {
    for (Long n = m; n <= degree; n++) {
      ConstIterator<Real> Pn  = leg_poly.begin() + N * ((degree * 2 - m + 1) * (m + 0) / 2 + n);
      ConstIterator<Real> Pn_ = leg_poly.begin() + N * ((degree * 2 - m + 0) * (m + 1) / 2 + n) * (m < n);
      Iterator     <Real> Hn  = poly_val.begin() + N * ((degree * 2 - m + 1) * (m + 0) / 2 + n);

      Real c2 = sqrt<Real>(m<n ? (n+m+1)*(n-m) : 0);
      for (Long i = 0; i < N; i++) {
        Real c1 = (X[i]*X[i]<1 ? m/sqrt<Real>(1-X[i]*X[i]) : 0);
        Hn[i] = c1*X[i]*Pn[i] + c2*Pn_[i];
      }
    }
  }
}


template <class Real> const Vector<Real>& SphericalHarmonics<Real>::LegendreNodes(Long p){
  assert(p<SCTL_SHMAXDEG);
  Vector<Real>& Qx=MatrixStore().Qx_[p];
  if(!Qx.Dim()){
    Vector<double> qx1(p+1);
    Vector<double> qw1(p+1);
    cgqf(p+1, 1, 0.0, 0.0, -1.0, 1.0, &qx1[0], &qw1[0]);
    assert(typeid(Real) == typeid(double) || typeid(Real) == typeid(float)); // TODO: works only for float and double
    if (Qx.Dim() != p+1) Qx.ReInit(p+1);
    for (Long i = 0; i < p + 1; i++) Qx[i] = -qx1[i];
  }
  return Qx;
}

template <class Real> const Vector<Real>& SphericalHarmonics<Real>::LegendreWeights(Long p){
  assert(p<SCTL_SHMAXDEG);
  Vector<Real>& Qw=MatrixStore().Qw_[p];
  if(!Qw.Dim()){
    Vector<double> qx1(p+1);
    Vector<double> qw1(p+1);
    cgqf(p+1, 1, 0.0, 0.0, -1.0, 1.0, &qx1[0], &qw1[0]);
    assert(typeid(Real) == typeid(double) || typeid(Real) == typeid(float)); // TODO: works only for float and double
    if (Qw.Dim() != p+1) Qw.ReInit(p+1);
    for (Long i = 0; i < p + 1; i++) Qw[i] = qw1[i];
  }
  return Qw;
}

template <class Real> const Vector<Real>& SphericalHarmonics<Real>::SingularWeights(Long p1){
  assert(p1<SCTL_SHMAXDEG);
  Vector<Real>& Sw=MatrixStore().Sw_[p1];
  if(!Sw.Dim()){
    const Vector<Real>& qx1 = LegendreNodes(p1);
    const Vector<Real>& qw1 = LegendreWeights(p1);

    std::vector<Real> Yf(p1+1,0);
    { // Set Yf
      Vector<Real> x0(1); x0=1.0;
      Vector<Real> alp0((p1+1)*(p1+2)/2);
      LegPoly(alp0, x0, p1);

      Vector<Real> alp((p1+1) * (p1+1)*(p1+2)/2);
      LegPoly(alp, qx1, p1);

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


template <class Real> const Matrix<Real>& SphericalHarmonics<Real>::MatFourier(Long p0, Long p1){
  assert(p0<SCTL_SHMAXDEG && p1<SCTL_SHMAXDEG);
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

template <class Real> const Matrix<Real>& SphericalHarmonics<Real>::MatFourierInv(Long p0, Long p1){
  assert(p0<SCTL_SHMAXDEG && p1<SCTL_SHMAXDEG);
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

template <class Real> const Matrix<Real>& SphericalHarmonics<Real>::MatFourierGrad(Long p0, Long p1){
  assert(p0<SCTL_SHMAXDEG && p1<SCTL_SHMAXDEG);
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


template <class Real> const FFT<Real>& SphericalHarmonics<Real>::OpFourier(Long Np){
  assert(Np<SCTL_SHMAXDEG);
  auto& Mf =MatrixStore().Mfftinv_ [Np];
  #pragma omp critical (SCTL_FFT_PLAN0)
  if(!Mf.Dim(0)){
    StaticArray<Long,1> fft_dim = {Np};
    Mf.Setup(FFT_Type::C2R, 1, Vector<Long>(1,fft_dim,false));
  }
  return Mf;
}

template <class Real> const FFT<Real>& SphericalHarmonics<Real>::OpFourierInv(Long Np){
  assert(Np<SCTL_SHMAXDEG);
  auto& Mf =MatrixStore().Mfft_ [Np];
  #pragma omp critical (SCTL_FFT_PLAN1)
  if(!Mf.Dim(0)){
    StaticArray<Long,1> fft_dim = {Np};
    Mf.Setup(FFT_Type::R2C, 1, Vector<Long>(1,fft_dim,false));
  }
  return Mf;
}


template <class Real> const std::vector<Matrix<Real>>& SphericalHarmonics<Real>::MatLegendre(Long p0, Long p1){
  assert(p0<SCTL_SHMAXDEG && p1<SCTL_SHMAXDEG);
  std::vector<Matrix<Real>>& Ml =MatrixStore().Ml_ [p0*SCTL_SHMAXDEG+p1];
  if(!Ml.size()){
    const Vector<Real>& qx1 = LegendreNodes(p1);
    Vector<Real> alp(qx1.Dim()*(p0+1)*(p0+2)/2);
    LegPoly(alp, qx1, p0);

    Ml.resize(p0+1);
    auto ptr = alp.begin();
    for(Long i=0;i<=p0;i++){
      Ml[i].ReInit(p0+1-i, qx1.Dim(), ptr);
      ptr+=Ml[i].Dim(0)*Ml[i].Dim(1);
    }
  }
  return Ml;
}

template <class Real> const std::vector<Matrix<Real>>& SphericalHarmonics<Real>::MatLegendreInv(Long p0, Long p1){
  assert(p0<SCTL_SHMAXDEG && p1<SCTL_SHMAXDEG);
  std::vector<Matrix<Real>>& Ml =MatrixStore().Mlinv_ [p0*SCTL_SHMAXDEG+p1];
  if(!Ml.size()){
    const Vector<Real>& qx1 = LegendreNodes(p0);
    const Vector<Real>& qw1 = LegendreWeights(p0);
    Vector<Real> alp(qx1.Dim()*(p1+1)*(p1+2)/2);
    LegPoly(alp, qx1, p1);

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
  return Ml;
}

template <class Real> const std::vector<Matrix<Real>>& SphericalHarmonics<Real>::MatLegendreGrad(Long p0, Long p1){
  assert(p0<SCTL_SHMAXDEG && p1<SCTL_SHMAXDEG);
  std::vector<Matrix<Real>>& Mdl=MatrixStore().Mdl_[p0*SCTL_SHMAXDEG+p1];
  if(!Mdl.size()){
    const Vector<Real>& qx1 = LegendreNodes(p1);
    Vector<Real> alp(qx1.Dim()*(p0+1)*(p0+2)/2);
    LegPolyDeriv(alp, qx1, p0);

    Mdl.resize(p0+1);
    auto ptr = alp.begin();
    for(Long i=0;i<=p0;i++){
      Mdl[i].ReInit(p0+1-i, qx1.Dim(), ptr);
      ptr+=Mdl[i].Dim(0)*Mdl[i].Dim(1);
    }
  }
  return Mdl;
}


template <class Real> void SphericalHarmonics<Real>::SHBasisEval(Long p0, const Vector<Real>& cos_theta_phi, Matrix<Real>& SHBasis) {
  Long M = (p0+1) * (p0+1);
  Long N = cos_theta_phi.Dim() / 2;
  assert(cos_theta_phi.Dim() == N * 2);

  Vector<Complex<Real>> exp_phi(N);
  Matrix<Real> LegP((p0+1)*(p0+2)/2, N);
  { // Set exp_phi, LegP
    Vector<Real> cos_theta(N);
    for (Long i = 0; i < N; i++) { // Set cos_theta, exp_phi
      cos_theta[i] = cos_theta_phi[i*2+0];
      exp_phi[i].real = cos(cos_theta_phi[i*2+1]);
      exp_phi[i].imag = sin(cos_theta_phi[i*2+1]);
    }

    Vector<Real> alp(LegP.Dim(0) * LegP.Dim(1), LegP.begin(), false);
    LegPoly(alp, cos_theta, p0);
  }

  { // Set SHBasis
    SHBasis.ReInit(N, M);
    Real s = 4 * sqrt<Real>(const_pi<Real>());
    for (Long k0 = 0; k0 < N; k0++) {
      Complex<Real> exp_phi_ = 1;
      Complex<Real> exp_phi1 = exp_phi[k0];
      for (Long m = 0; m <= p0; m++) {
        for (Long n = m; n <= p0; n++) {
          Long poly_idx = (2 * p0 - m + 1) * m / 2 + n;
          Long basis_idx = (2 * p0 - m + 2) * m - (m ? p0+1 : 0) + n;
          SHBasis[k0][basis_idx] = LegP[poly_idx][k0] * exp_phi_.real * s;
          if (m) { // imaginary part
            basis_idx += (p0+1-m);
            SHBasis[k0][basis_idx] = -LegP[poly_idx][k0] * exp_phi_.imag * s;
          } else {
            SHBasis[k0][basis_idx] = SHBasis[k0][basis_idx] * 0.5;
          }
        }
        exp_phi_ = exp_phi_ * exp_phi1;
      }
    }
  }
  assert(SHBasis.Dim(0) == N);
  assert(SHBasis.Dim(1) == M);
}

template <class Real> void SphericalHarmonics<Real>::VecSHBasisEval(Long p0, const Vector<Real>& cos_theta_phi, Matrix<Real>& SHBasis) {
  Long M = (p0+1) * (p0+1);
  Long N = cos_theta_phi.Dim() / 2;
  assert(cos_theta_phi.Dim() == N * 2);

  Long p_ = p0 + 1;
  Long M_ = (p_+1) * (p_+1);
  Matrix<Real> Ynm(N, M_);
  SHBasisEval(p_, cos_theta_phi, Ynm);

  Vector<Real> cos_theta(N);
  for (Long i = 0; i < N; i++) { // Set cos_theta
    cos_theta[i] = cos_theta_phi[i*2+0];
  }

  { // Set SHBasis
    SHBasis.ReInit(N * COORD_DIM, COORD_DIM * M);
    SHBasis = 0;
    const Complex<Real> imag(0,1);
    for (Long i = 0; i < N; i++) {
      Real s = 1 / sqrt<Real>(1 - cos_theta[i] * cos_theta[i]);
      for (Long m = 0; m <= p0; m++) {
        for (Long n = m; n <= p0; n++) {
          auto Y = [&](Long n, Long m) {
            Complex<Real> c;
            if (0 <= m && m <= n && n <= p_) {
              Long idx = (2 * p_ - m + 2) * m - (m ? p_+1 : 0) + n;
              c.real = Ynm[i][idx];
              if (m) {
                idx += (p_+1-m);
                c.imag = Ynm[i][idx];
              }
            }
            return c;
          };
          auto write_coeff = [&](Complex<Real> c, Long n, Long m, Long k0, Long k1) {
            if (0 <= m && m <= n && n <= p0 && 0 <= k0 && k0 < COORD_DIM && 0 <= k1 && k1 < COORD_DIM) {
              Long idx = (2 * p0 - m + 2) * m - (m ? p0+1 : 0) + n;
              SHBasis[i * COORD_DIM + k1][k0 * M + idx] = c.real;
              if (m) {
                idx += (p0+1-m);
                SHBasis[i * COORD_DIM + k1][k0 * M + idx] = c.imag;
              }
            }
          };

          auto A = [&](Long n, Long m) { return (0<=n && m<=n && n<=p_ ? sqrt<Real>(n*n * ((n+1)*(n+1) - m*m) / (Real)((2*n+1)*(2*n+3))) : 0); };
          auto B = [&](Long n, Long m) { return (0<=n && m<=n && n<=p_ ? sqrt<Real>((n+1)*(n+1) * (n*n - m*m) / (Real)((2*n+1)*(2*n-1))) : 0); };
          Complex<Real> AYBY = A(n,m) * Y(n+1,m) - B(n,m) * Y(n-1,m);

          Complex<Real> Fv2r = Y(n,m) * (-n-1);
          Complex<Real> Fw2r = Y(n,m) * n;
          Complex<Real> Fx2r = 0;

          Complex<Real> Fv2t = AYBY * s;
          Complex<Real> Fw2t = AYBY * s;
          Complex<Real> Fx2t = imag * m * Y(n,m) * s;

          Complex<Real> Fv2p = -imag * m * Y(n,m) * s;
          Complex<Real> Fw2p = -imag * m * Y(n,m) * s;
          Complex<Real> Fx2p = AYBY * s;

          write_coeff(Fv2r, n, m, 0, 0);
          write_coeff(Fw2r, n, m, 1, 0);
          write_coeff(Fx2r, n, m, 2, 0);

          write_coeff(Fv2t, n, m, 0, 1);
          write_coeff(Fw2t, n, m, 1, 1);
          write_coeff(Fx2t, n, m, 2, 1);

          write_coeff(Fv2p, n, m, 0, 2);
          write_coeff(Fw2p, n, m, 1, 2);
          write_coeff(Fx2p, n, m, 2, 2);
        }
      }
    }
  }
  assert(SHBasis.Dim(0) == N * COORD_DIM);
  assert(SHBasis.Dim(1) == COORD_DIM * M);
}


template <class Real> const std::vector<Matrix<Real>>& SphericalHarmonics<Real>::MatRotate(Long p0){
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
  std::vector<Matrix<Real>>& Mr=MatrixStore().Mr_[p0];
  if(!Mr.size()){
    const Real SQRT2PI=sqrt(2*M_PI);
    Long Ncoef=p0*(p0+2);
    Long Ngrid=2*p0*(p0+1);
    Long Naleg=(p0+1)*(p0+2)/2;

    Matrix<Real> Mcoord0(3,Ngrid);
    const Vector<Real>& x=LegendreNodes(p0);
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
        const Vector<Real> Vcoord1(Mcoord1.Dim(0)*Mcoord1.Dim(1), Mcoord1.begin(), false);
        Vector<Real> Vleg(Mleg.Dim(0)*Mleg.Dim(1), Mleg.begin(), false);
        LegPoly(Vleg, Vcoord1, p0);
      }

      Vector<Real> theta(Ngrid);
      for(Long i=0;i<theta.Dim();i++){ // Set theta
        theta[i]=atan2(Mcoord1[1][i],Mcoord1[2][i]); // TODO: works only for float and double
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
      Grid2SHC(Vcoef2grid, p0+1, 2*p0, p0, Vcoef2coef, SHCArrange::COL_MAJOR_NONZERO);

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

  Vector<Real> B0, B1;
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
      const Matrix<Real> Min (i1-i0,2*p0, (Iterator<Real>)X.begin()+i0*2*p0, false);
      Matrix<Real> Mout(i1-i0,2*p1, B2.begin(), false);
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

template <class Real> void SphericalHarmonics<Real>::RotateAll(const Vector<Real>& S, Long p0, Long dof, Vector<Real>& S_){
  const std::vector<Matrix<Real>>& Mr=MatRotate(p0);
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
  const Matrix<Real> S0(N*dof, Ncoef, (Iterator<Real>)S.begin(), false);
  Matrix<Real> S1(N*dof*p0*(p0+1), Ncoef, S_.begin(), false);

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
  Matrix<Real> S0(N*dof*p0*(p0+1), Ncoef, S.begin(), false);
  const Matrix<Real> S1(N*dof*p0*(p0+1), Ncoef, (Iterator<Real>)S_.begin(), false);

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
          const Matrix<Real> Min(p0*dof, Ncoef, (Iterator<Real>)S1[idx0], false);
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

    Vector<Real> _SLMatrix, _DLMatrix;
    if(SLMatrix) _SLMatrix.ReInit((b-a)*(Ncoef*COORD_DIM)*(Ncoef*COORD_DIM), SLMatrix->begin()+a*(Ncoef*COORD_DIM)*(Ncoef*COORD_DIM), false);
    if(DLMatrix) _DLMatrix.ReInit((b-a)*(Ncoef*COORD_DIM)*(Ncoef*COORD_DIM), DLMatrix->begin()+a*(Ncoef*COORD_DIM)*(Ncoef*COORD_DIM), false);
    const Vector<Real> _S        ((b-a)*(Ngrid*COORD_DIM)                  , (Iterator<Real>)S.begin()+a*(Ngrid*COORD_DIM), false);

    if(SLMatrix && DLMatrix) StokesSingularInteg_< true,  true>(_S, p0, p1, _SLMatrix, _DLMatrix);
    else        if(SLMatrix) StokesSingularInteg_< true, false>(_S, p0, p1, _SLMatrix, _DLMatrix);
    else        if(DLMatrix) StokesSingularInteg_<false,  true>(_S, p0, p1, _SLMatrix, _DLMatrix);
  }
}

template <class Real> template <bool SLayer, bool DLayer> void SphericalHarmonics<Real>::StokesSingularInteg_(const Vector<Real>& X0, Long p0, Long p1, Vector<Real>& SL, Vector<Real>& DL){

  Profile::Tic("Rotate");
  Vector<Real> S0, S;
  SphericalHarmonics<Real>::Grid2SHC(X0, p0+1, 2*p0, p0, S0, SHCArrange::COL_MAJOR_NONZERO);
  SphericalHarmonics<Real>::RotateAll(S0, p0, COORD_DIM, S);
  Profile::Toc();


  Profile::Tic("Upsample");
  Vector<Real> X, X_theta, X_phi, trg;
  SphericalHarmonics<Real>::SHC2Grid(S, SHCArrange::COL_MAJOR_NONZERO, p0, p1+1, 2*p1, &X, &X_theta, &X_phi);
  SphericalHarmonics<Real>::SHC2Pole(S, SHCArrange::COL_MAJOR_NONZERO, p0, trg);
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
    const Vector<Real>& qw=SphericalHarmonics<Real>::SingularWeights(p1);

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
                Real x_theta=X_phi[(i*COORD_DIM+0)*M1+s];
                Real y_theta=X_phi[(i*COORD_DIM+1)*M1+s];
                Real z_theta=X_phi[(i*COORD_DIM+2)*M1+s];

                Real x_phi=X_theta[(i*COORD_DIM+0)*M1+s];
                Real y_phi=X_theta[(i*COORD_DIM+1)*M1+s];
                Real z_phi=X_theta[(i*COORD_DIM+2)*M1+s];

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
  Vector<Real> SL1, DL1;
  SphericalHarmonics<Real>::SHC2GridTranspose(SL0, p1, p0, SL1);
  SphericalHarmonics<Real>::SHC2GridTranspose(DL0, p1, p0, DL1);
  Profile::Toc();


  Profile::Tic("RotateTranspose");
  Vector<Real> SL2, DL2;
  SphericalHarmonics<Real>::RotateTranspose(SL1, p0, 2*6, SL2);
  SphericalHarmonics<Real>::RotateTranspose(DL1, p0, 2*6, DL2);
  Profile::Toc();


  Profile::Tic("Rearrange");
  Vector<Real> SL3, DL3;
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
  SphericalHarmonics<Real>::Grid2SHC(SL3, p0+1, 2*p0, p0, SL, SHCArrange::COL_MAJOR_NONZERO);
  SphericalHarmonics<Real>::Grid2SHC(DL3, p0+1, 2*p0, p0, DL, SHCArrange::COL_MAJOR_NONZERO);
  Profile::Toc();

}

}  // end namespace
