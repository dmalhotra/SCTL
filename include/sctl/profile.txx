#ifndef _SCTL_PROFILE_TXX_
#define _SCTL_PROFILE_TXX_

#include <sstream>            // for stringstream
#include <omp.h>              // for omp_get_wtime
#include <stdio.h>            // for size_t, sprintf
#include <algorithm>          // for max
#include <array>              // for array
#include <atomic>             // for atomic, memory_order
#include <iomanip>            // for operator<<, setw
#include <iostream>           // for basic_ostream, operator<<, cout, left
#include <map>                // for map
#include <sstream>            // for basic_stringstream
#include <stack>              // for stack
#include <string>             // for basic_string, allocator, char_traits
#include <vector>             // for vector

#include "sctl/common.hpp"    // for Long, SCTL_ASSERT, Integer, SCTL_ASSERT...
#include SCTL_INCLUDE(profile.hpp)   // for Profile, ProfileCounter, operator*, ope...
#include SCTL_INCLUDE(comm.hpp)      // for CommOp, Comm
#include SCTL_INCLUDE(comm.txx)      // for Comm::Rank, Comm::Size, Comm::Allreduce
#include SCTL_INCLUDE(iterator.hpp)  // for ConstIterator, Iterator
#include SCTL_INCLUDE(iterator.txx)  // for Ptr2ConstItr, Ptr2Itr, ConstIterator::C...

namespace SCTL_NAMESPACE {

  template <class E> class Profile::ExprWrapper {
    public:

      std::vector<double> operator()(const std::vector<double>& counters, const Comm* comm) const {
        return ((const E*)this)->operator()(counters, comm);
      }

      static void* CopyInstance(const void* self) {
        E* expr = new E(*(E*)self);
        return expr;
      }

      static std::vector<double> Eval(const std::vector<double>& counters, const Comm* comm, const void* self) {
        return ((const E*)self)->operator()(counters, comm);
      }

      static void DeleteInstance(void** self_ptr) {
        SCTL_ASSERT(self_ptr);
        if (!*self_ptr) return;
        delete (E*)(*self_ptr);
        (*self_ptr) = nullptr;
      }
  };

  template <> class Profile::ExprWrapper<void> {
    public:

      inline ExprWrapper() : instance(nullptr) {}

      template <class E> ExprWrapper(const ExprWrapper<E>& e) : instance(nullptr) {
        Copy(e);
      }

      inline ExprWrapper(const ExprWrapper<void>& e) : instance(nullptr) {
        Copy(e);
      }

      template <class E> ExprWrapper<void>& operator=(const ExprWrapper<E>& e) {
        Copy(e);
        return *this;
      }

      inline ExprWrapper<void>& operator=(const ExprWrapper<void>& e) {
        Copy(e);
        return *this;
      }

      inline ~ExprWrapper() {
        if (instance) del_fn(&instance);
      }


      inline std::vector<double> operator()(const std::vector<double>& counters, const Comm* comm) const {
        return eval_fn(counters, comm, instance);
      }

      inline static void* CopyInstance(const void* self) {
        ExprWrapper<void>* instance_ptr = new ExprWrapper<void>;
        const auto self_ = (ExprWrapper<void>*)self;
        instance_ptr->copy_fn = self_->copy_fn;
        instance_ptr->eval_fn = self_->eval_fn;
        instance_ptr->del_fn = self_->del_fn;
        instance_ptr->instance = self_->copy_fn(self_->instance);
        return instance_ptr;
      }

      inline static std::vector<double> Eval(const std::vector<double>& counters, const Comm* comm, const void* self) {
        const auto self_ = (ExprWrapper<void>*)self;
        return self_->eval_fn(counters, comm, self_->instance);
      }

      inline static void DeleteInstance(void** self_ptr) {
        SCTL_ASSERT(self_ptr);
        if (!*self_ptr) return;
        delete (ExprWrapper<void>*)*self_ptr;
        (*self_ptr) = nullptr;
      }

    private:

      template <class Expr> void Copy(const Expr& e) {
        if (instance) del_fn(&instance);
        copy_fn = Expr::CopyInstance;
        eval_fn = Expr::Eval;
        del_fn = Expr::DeleteInstance;
        instance = copy_fn(&e);
      }

      inline void Copy(const ExprWrapper<void>& e) {
        if (instance) del_fn(&instance);
        copy_fn = e.copy_fn;
        eval_fn = e.eval_fn;
        del_fn = e.del_fn;
        instance = copy_fn(e.instance);
      }

      void* (*copy_fn)(const void*);
      std::vector<double> (*eval_fn)(const std::vector<double>&, const Comm*, const void*);
      void (*del_fn)(void**);
      void* instance;
  };

  class Profile::ExprScalar : public Profile::ExprWrapper<ExprScalar> {
    public:

      ExprScalar() = default;

      inline ExprScalar(ProfileCounter field) : field_((Long)field), value_(0) {}

      inline ExprScalar(double value) : field_((Long)ProfileCounter::FIELD_COUNT), value_(value) {}

      inline std::vector<double> operator()(const std::vector<double>& counters, const Comm* comm) const {
        const Long Nfield = (Long)ProfileCounter::FIELD_COUNT;
        const Long N = counters.size() / Nfield;
        std::vector<double> val_vec(N);
        if (field_ == Nfield) {
          for (Long i = 0; i < N; i++) {
            val_vec[i] = value_;
          }
        } else {
          for (Long i = 0; i < N; i++) {
            val_vec[i] = counters[i*Nfield + field_];
          }
        }
        return val_vec;
      }

    private:
      Long field_;
      double value_;
  };

  template <class UnaryOp> class Profile::ExprUnary : public Profile::ExprWrapper<ExprUnary<UnaryOp>> {
    public:

      ExprUnary() = default;

      ExprUnary(const ExprWrapper<void>& e1, const UnaryOp& op) : e1_(e1), op_(op) {}

      std::vector<double> operator()(const std::vector<double>& counters, const Comm* comm) const {
        return op_(e1_(counters, comm), comm);
      }

    private:
      ExprWrapper<void> e1_;
      UnaryOp op_;
  };

  template <class BinaryOp> class Profile::ExprBinary : public Profile::ExprWrapper<ExprBinary<BinaryOp>> {
    public:

      ExprBinary() = default;

      ExprBinary(const ExprWrapper<void>& e1, const ExprWrapper<void>& e2, const BinaryOp& op) : e1_(e1), e2_(e2), op_(op) {}

      std::vector<double> operator()(const std::vector<double>& counters, const Comm* comm) const {
        return op_(e1_(counters, comm), e2_(counters, comm), comm);
      }

    private:
      ExprWrapper<void> e1_, e2_;
      BinaryOp op_;
  };



  inline Profile::ProfExpr operator+(const Profile::ProfExpr& u, const Profile::ProfExpr& v) {
    const auto op = [](std::vector<double> x, const std::vector<double>& y, const Comm* comm) {
      SCTL_ASSERT(x.size() == y.size());
      for (Long i = 0; i < (Long)x.size(); i++) x[i] += y[i];
      return x;
    };
    return Profile::ExprBinary<decltype(op)>(u, v, op);
  }

  inline Profile::ProfExpr operator-(const Profile::ProfExpr& u, const Profile::ProfExpr& v) {
    const auto op = [](std::vector<double> x, const std::vector<double>& y, const Comm* comm) {
      SCTL_ASSERT(x.size() == y.size());
      for (Long i = 0; i < (Long)x.size(); i++) x[i] -= y[i];
      return x;
    };
    return Profile::ExprBinary<decltype(op)>(u, v, op);
  }

  inline Profile::ProfExpr operator*(const Profile::ProfExpr& u, const Profile::ProfExpr& v) {
    const auto op = [](std::vector<double> x, const std::vector<double>& y, const Comm* comm) {
      SCTL_ASSERT(x.size() == y.size());
      for (Long i = 0; i < (Long)x.size(); i++) x[i] *= y[i];
      return x;
    };
    return Profile::ExprBinary<decltype(op)>(u, v, op);
  }

  inline Profile::ProfExpr operator/(const Profile::ProfExpr& u, const Profile::ProfExpr& v) {
    const auto op = [](std::vector<double> x, const std::vector<double>& y, const Comm* comm) {
      SCTL_ASSERT(x.size() == y.size());
      for (Long i = 0; i < (Long)x.size(); i++) x[i] /= y[i];
      return x;
    };
    return Profile::ExprBinary<decltype(op)>(u, v, op);
  }

  inline Profile::ProfExpr operator*(const Profile::ProfExpr& u, const double a) {
    const auto op = [a](std::vector<double> x, const Comm* comm) {
      for (Long i = 0; i < (Long)x.size(); i++) x[i] *= a;
      return x;
    };
    return Profile::ExprUnary<decltype(op)>(u, op);
  }



  struct Profile::ProfileData {
    const double t0;
    bool enable_state;

    std::stack<int> verb;
    std::stack<bool> sync;
    std::stack<std::string> name;
    std::stack<const Comm*> comm;

    std::vector<bool> e_log;
    std::vector<std::string> n_log;
    std::vector<double> counter_log;

    std::array<std::atomic<Long>, Nfield> counters;
    std::map<std::string, ProfExpr> prof_fields;

    inline ProfileData() : t0(omp_get_wtime()), enable_state(false) {
      constexpr double gb_scale = (1./1024/1024/1024);
      for (auto& x : counters) x = 0;

      e_log.reserve(1e5);
      n_log.reserve(1e5);
      for (auto& name : n_log) name.reserve(20);
      counter_log.reserve((Long)(1e5 * Nfield));

      prof_fields["0"]  = ExprScalar(0.);
      prof_fields["t"]  = ExprScalar(ProfileCounter::TIME) * 1e-9;
      prof_fields["f"] = ExprScalar(ProfileCounter::FLOP) * 1e-9;
      prof_fields["f/s"] = prof_fields["f"] / prof_fields["t"];

      prof_fields["alloc_count"] = ExprScalar(ProfileCounter::HEAP_ALLOC_COUNT);
      prof_fields["alloc_m"]     = ExprScalar(ProfileCounter::HEAP_ALLOC_BYTES) * gb_scale;
      prof_fields["free_count"]  = ExprScalar(ProfileCounter::HEAP_FREE_COUNT);
      prof_fields["free_m" ]     = ExprScalar(ProfileCounter::HEAP_FREE_BYTES) * gb_scale;
      prof_fields["m"]           = prof_fields["alloc_m"] - prof_fields["free_m"];

      prof_fields["comm_count"]      = ExprScalar(ProfileCounter::PROF_MPI_COUNT);
      prof_fields["comm_m"]          = ExprScalar(ProfileCounter::PROF_MPI_BYTES) * gb_scale;
      prof_fields["comm_coll_m"]     = ExprScalar(ProfileCounter::PROF_MPI_COLLECTIVE_BYTES) * gb_scale;
      prof_fields["comm_coll_count"] = ExprScalar(ProfileCounter::PROF_MPI_COLLECTIVE_COUNT);

      prof_fields["custom1"] = ExprScalar(ProfileCounter::CUSTOM1);
      prof_fields["custom2"] = ExprScalar(ProfileCounter::CUSTOM2);
      prof_fields["custom3"] = ExprScalar(ProfileCounter::CUSTOM3);
      prof_fields["custom4"] = ExprScalar(ProfileCounter::CUSTOM4);
      prof_fields["custom5"] = ExprScalar(ProfileCounter::CUSTOM5);

      const auto comm_size_fn = [](std::vector<double> v, const Comm* comm) {
        const Long np = (comm ? comm->Size() : 1);
        for (auto& x : v) x = (double)np;
        return v;
      };
      prof_fields["comm_size"] = Profile::ExprUnary<decltype(comm_size_fn)>(ExprScalar(0.), comm_size_fn);

      prof_fields["t_min"] = Profile::CommReduceExpr(prof_fields["t"], CommOp::MIN);
      prof_fields["t_max"] = Profile::CommReduceExpr(prof_fields["t"], CommOp::MAX);
      prof_fields["t_avg"] = Profile::CommReduceExpr(prof_fields["t"], CommOp::SUM) / prof_fields["comm_size"];

      prof_fields["f_min"] = Profile::CommReduceExpr(prof_fields["f"], CommOp::MIN);
      prof_fields["f_max"] = Profile::CommReduceExpr(prof_fields["f"], CommOp::MAX);
      prof_fields["f_avg"] = Profile::CommReduceExpr(prof_fields["f"], CommOp::SUM) / prof_fields["comm_size"];
      prof_fields["f_total"] = Profile::CommReduceExpr(prof_fields["f"], CommOp::SUM);

      prof_fields["f/s_min"] = Profile::CommReduceExpr(prof_fields["f"] / prof_fields["t"], CommOp::MIN);
      prof_fields["f/s_max"] = Profile::CommReduceExpr(prof_fields["f"] / prof_fields["t"], CommOp::MAX);
      prof_fields["f/s_avg"] = Profile::CommReduceExpr(prof_fields["f"] / prof_fields["t"], CommOp::SUM) / prof_fields["comm_size"];
      prof_fields["f/s_total"] = prof_fields["f_total"] / prof_fields["t_max"];

      prof_fields["m_min"] = Profile::CommReduceExpr(prof_fields["m"], CommOp::MIN);
      prof_fields["m_max"] = Profile::CommReduceExpr(prof_fields["m"], CommOp::MAX);
      prof_fields["m_avg"] = Profile::CommReduceExpr(prof_fields["m"], CommOp::SUM) / prof_fields["comm_size"];
    }
  };

  inline Profile::ProfileData& Profile::GetProfData() {
    static ProfileData p;
    return p;
  }



  inline const Profile::ProfExpr Profile::GetProfField(const std::string& name) {
    if (!GetProfData().prof_fields.count(name)) {
      SCTL_WARN("Unknown profile field name ignored:"<<name);
      return ExprScalar(0.);
    }
    return GetProfData().prof_fields[name];
  }

  inline void Profile::SetProfField(const std::string& name, const Profile::ProfExpr& expr) {
    GetProfData().prof_fields[name] = expr;
  }

  template <class UnaryOp> Profile::ProfExpr Profile::UnaryExpr(const Profile::ProfExpr& e1, const UnaryOp& op) {
    return Profile::ExprUnary<decltype(op)>(e1, op);
  }

  template <class BinaryOp> Profile::ProfExpr Profile::BinaryExpr(const Profile::ProfExpr& e1, const Profile::ProfExpr& e2, const BinaryOp& op) {
    return Profile::ExprBinary<decltype(op)>(e1, e2, op);
  }

  inline Profile::ProfExpr Profile::CommReduceExpr(const Profile::ProfExpr& u, const CommOp comm_op) {
    const auto op = [comm_op](std::vector<double> x, const Comm* comm) {
      const Long N = (Long)x.size();
      if (!comm || !N) return x;
      std::vector<double> y(N);
      comm->Allreduce(Ptr2ConstItr<double>(&x[0],N), Ptr2Itr<double>(&y[0],N), N, comm_op);
      return y;
    };
    return Profile::ExprUnary<decltype(op)>(u, op);
  }




  inline bool Profile::Enable(bool state) {
    const bool orig_val = GetProfData().enable_state;
    GetProfData().enable_state = state;
    return orig_val;
  }

  inline void Profile::print(const Comm* comm_, std::vector<std::string> field_lst, std::vector<std::string> format_lst) {
    if (!field_lst.size()) {
      if (!comm_ || comm_->Size()==1) field_lst = {"t", "f", "f/s", "m"};
      else field_lst = {"t_avg", "t_max", "f_avg", "f_max", "f/s_min", "f/s_avg", "f/s_total", "m_max"};
    }
    if (format_lst.size() != field_lst.size()) {
      const Long N = field_lst.size() - format_lst.size();
      for (Long i = 0; i < N; i++) format_lst.push_back("%.4f");
    }
    const Long Ncolumn = field_lst.size();

    std::vector<std::string> name; // row label
    std::vector<Integer> depth; // nesting depth
    std::vector<double> counter; // data counters
    Long name_max_length = 0; // max length of name-column (first column)
    { // name, name_max_length, depth, counter
      std::stack<Long> idx_stack;
      ProfileData& prof = GetProfData();
      for (Long i = prof.e_log.size()-1; i >= 0; i--) {
        if (prof.e_log[i]==0) {
          idx_stack.push(i);
        } else {
          if (!idx_stack.size()) {
            //prof.e_log.resize(i+1);
            //prof.n_log.resize(i+1);
            //prof.counter_log.resize((i+1)*Nfield);
            break;
          }
          const Long i0 = idx_stack.top();

          const auto name_ = prof.n_log[i];
          const auto depth_ = idx_stack.size()-1;
          name.push_back(name_);
          depth.push_back(depth_);
          name_max_length = std::max<Long>(name_max_length, name_.size() + depth_*2 + 2);
          for (Long k = 0; k < Nfield; k++) {
            counter.push_back(prof.counter_log[i0*Nfield+k] - prof.counter_log[i*Nfield+k]);
          }

          idx_stack.pop();
        }
      }
    }
    if (!name.size()) return;

    std::vector<Long> column_width(Ncolumn);
    std::vector<std::vector<std::string>> column_lst(Ncolumn);
    for (Long i = 0; i < Ncolumn; i++) { // Set column_lst, column_width
      const auto column = GetProfField(field_lst[i])(counter, comm_);
      Long max_width = field_lst[i].size();

      const Long N = column.size();
      std::vector<std::string> column_str(N);
      for (Long j = 0; j < N; j++) {
        char buffer[100];
        sprintf(buffer, format_lst[i].c_str(), column[j]);
        column_str[j] = std::string(buffer);
        max_width = std::max(max_width, (Long)column_str[j].size());
      }
      column_lst[i] = column_str;
      column_width[i] = max_width;
    }

    std::string out_str;
    std::string tree_str;
    for (Long i = 0; i < (Long)name.size(); i++) { // build table bottom up
      // Set tree_str (the tree structure)
      for (Long k = tree_str.size()/2; k < depth[i]; k++) tree_str = tree_str + "  ";
      if (i > 0 && depth[i] > depth[i-1]) tree_str[depth[i-1]*2] = '|';
      tree_str.resize(depth[i]*2);

      std::stringstream row_ss(std::stringstream::in | std::stringstream::out);
      row_ss << tree_str << "+-" << std::setw(name_max_length-tree_str.size()-2) << std::left << name[i]; // print name
      for (Long j = 0; j < Ncolumn; j++) row_ss << "  " << std::setw(column_width[j]) << std::right << column_lst[j][i]; // print values
      row_ss << '\n';

      if (i > 0 && depth[i-1] < depth[i]) row_ss << tree_str << '\n'; // add separator row
      out_str = row_ss.str() + out_str; // prepend row_ss to out_str
    }
    { // Print column headers
      std::stringstream ss(std::stringstream::in | std::stringstream::out);
      ss << std::string(name_max_length, ' ');
      for (Long j = 0; j < Ncolumn; j++) ss << "  " << std::setw(column_width[j]) << field_lst[j];
      out_str = ss.str() + '\n' + out_str;
    }
    if (comm_==nullptr || !comm_->Rank()) std::cout<<out_str<<'\n';
  }

  inline void Profile::reset() {
    ProfileData& prof = GetProfData();
    while (!prof.verb.empty()) prof.verb.pop();
    while (!prof.sync.empty()) prof.sync.pop();
    while (!prof.name.empty()) prof.name.pop();
    while (!prof.comm.empty()) prof.comm.pop();

    prof.e_log.clear();
    prof.n_log.clear();
    prof.counter_log.clear();
  }


  #if SCTL_PROFILE >= 0

  inline Long Profile::IncrementCounter(const ProfileCounter prof_field, const Long x) {
    return GetProfData().counters[(Long)prof_field].fetch_add(x,std::memory_order_relaxed);
  }

  inline void Profile::Tic(const char* name_, const Comm* comm_, bool sync_, Integer verbose) {
    ProfileData& prof = GetProfData();
    if (!prof.enable_state) return;
    // sync_=true;

    prof.verb.push((prof.verb.size()?prof.verb.top():0) + verbose);
    if (prof.verb.top() <= SCTL_PROFILE) {
      if (comm_ != nullptr && sync_) comm_->Barrier();
  #ifdef SCTL_VERBOSE
      if (comm_?!comm_->Rank():1) {
        for (size_t i = 0; i < prof.name.size(); i++) std::cout << "    ";
        std::cout << "\033[1;31m" << name_ << "\033[0m {\n";
      }
  #endif
      prof.name.push(name_);
      prof.comm.push(comm_);
      prof.sync.push(sync_);

      prof.e_log.push_back(true);
      prof.n_log.push_back(prof.name.top());
      prof.counters[(Long)ProfileCounter::TIME].store((Long)((omp_get_wtime()-prof.t0)*1e9),std::memory_order_relaxed);
      for (Long i = 0; i < Nfield; i++) prof.counter_log.push_back(prof.counters[i]);
    }
  }

  inline void Profile::Toc() {
    ProfileData& prof = GetProfData();
    if (!prof.enable_state) return;
    SCTL_ASSERT_MSG(!prof.verb.empty(), "Unbalanced extra Toc()");

    if (prof.verb.top() <= SCTL_PROFILE) {
      SCTL_ASSERT_MSG(!prof.name.empty(), "Unbalanced extra Toc()");

      const std::string& name_ = prof.name.top();
      const Comm* comm_ = prof.comm.top();
      const bool sync_ = prof.sync.top();
      SCTL_UNUSED(sync_);

      prof.e_log.push_back(false);
      prof.n_log.push_back(name_);
      prof.counters[(Long)ProfileCounter::TIME].store((Long)((omp_get_wtime()-prof.t0)*1e9),std::memory_order_relaxed);
      for (Long i = 0; i < Nfield; i++) prof.counter_log.push_back(prof.counters[i]);

  #ifndef NDEBUG
      if (comm_ != nullptr && sync_) comm_->Barrier();
  #endif
      prof.name.pop();
      prof.comm.pop();
      prof.sync.pop();

  #ifdef SCTL_VERBOSE
      if (comm_?!comm_->Rank():1) {
        for (size_t i = 0; i < prof.name.size(); i++) std::cout << "    ";
        std::cout << "}\n";
      }
  #endif
    }
    prof.verb.pop();
  }

  #else

  inline Long Profile::IncrementCounter(const ProfileCounter prof_field, const Long x) { return 0; }

  inline void Profile::Tic(const char* name_, const Comm* comm_, bool sync_, Integer verbose) {}

  inline void Profile::Toc() {}

  #endif

  inline Profile::Scoped::Scoped(const char* name_, const Comm* comm_, bool sync_, Integer verbose) {
    Profile::Tic(name_, comm_, sync_, verbose);
  }

  inline Profile::Scoped::~Scoped() {
    Profile::Toc();
  }


}  // end namespace

#endif // _SCTL_PROFILE_TXX_
