#ifndef _SCTL_PROFILE_HPP_
#define _SCTL_PROFILE_HPP_

#include <string>            // for basic_string, string
#include <vector>            // for vector

#include "sctl/common.hpp"   // for Long, Integer, sctl

#ifndef SCTL_PROFILE
#define SCTL_PROFILE -1
#endif

namespace sctl {

class Comm;
enum class CommOp;

/**
 * Counters available to use with the Profile class.
 */
enum class ProfileCounter: Long {
  TIME,
  FLOP,
  HEAP_ALLOC_COUNT,
  HEAP_ALLOC_BYTES,
  HEAP_FREE_COUNT,
  HEAP_FREE_BYTES,
  PROF_MPI_BYTES,
  PROF_MPI_COUNT,
  PROF_MPI_COLLECTIVE_BYTES,
  PROF_MPI_COLLECTIVE_COUNT,
  CUSTOM1,
  CUSTOM2,
  CUSTOM3,
  CUSTOM4,
  CUSTOM5,
  FIELD_COUNT
};

/**
 * This class allows for manual instrumentation of code. It allows the user to define profiling blocks. It then reports
 * the change in values of various counters (time, flops, memory allocations, etc) between the start and end of each
 * profiling block.
 */
class Profile {
  static constexpr Long Nfield = (Long)ProfileCounter::FIELD_COUNT;

  template <typename E> class ExprWrapper;
  using ProfExpr = ExprWrapper<void>;

 public:

  /**
   * Enable or disable profiler.
   *
   * @return the last state of the profiler.
   */
  static bool Enable(bool state);

  /**
   * Marks the start of a profiling block.
   *
   * @param[in] name name for the profiling block.
   *
   * @param[in] comm_ptr pointer to Comm object (can be nullptr).
   *
   * @param[in] sync whether to synchronize (using Comm::Barrier()) before the
   * profiling block.
   *
   * @param[in] verbose whether to display profiling block on stdout.
   */
  static void Tic(const char* name, const Comm* comm_ptr = nullptr, bool sync = false, Integer verbose = 1);

  /**
   * Marks the end of a profiling block.
   */
  static void Toc();

  /**
   * Defines a profiling block through the lifetime/scope of its instance.
   */
  struct Scoped {
    /**
     * Constructor.
     *
     * @param[in] name name for the profiling block.
     *
     * @param[in] comm_ptr pointer to Comm object (can be nullptr).
     *
     * @param[in] sync whether to synchronize (using Comm::Barrier()) before the profiling block.
     *
     * @param[in] verbose whether to display profiling block on stdout.
     */
    Scoped(const char* name_, const Comm* comm_ = nullptr, bool sync_ = false, Integer verbose = 1);
    ~Scoped();

    Scoped() = delete;
    Scoped(const Scoped&) = delete;
    Scoped& operator= (const Scoped&) = delete;
  };


  /**
   * Increment a profiling counter.
   *
   * @return the last value of the counter.
   *
   * @note ProfileCounter::TIME (ns units) must not be changed.
   */
  static Long IncrementCounter(const ProfileCounter prof_field, const Long x);

  /**
   * Returns a profiling expression identified by a string name.
   *
   * @param[in] name string identifier for a profiling expression.
   * Predefined string values corresponding to each ProfileCounter are:
   * t : TIME (in s)
   * f : FLOP (in GFLOPs)
   * alloc_count : number of heap allocations
   * alloc_m     : amount of heap memory allocated (in GB)
   * free_count  : number of times heap freed
   * free_m      : amount of heap memory freed (in GB)
   * comm_count      : number of Comm point-to-point communications
   * comm_m          : amount of memory transferred in p-to-p comm (in GB)
   * comm_coll_count : number of collective communications
   * comm_coll_m     : amount of memory transferred in collective comm (in GB)
   * custom1, custom2, custom3, custom4, custom5 : custom counters
   *
   * The following additional and derived quantities are also predefined:
   * 0   : constant zero value
   * f/s : FLOP/TIME (in GFLOP/s)
   * m   : change in heap memory (in GB)
   * comm_size : constant value Comm::Size()
   *
   * Also defined are *_min (minimum), *_max (maximum), and *_avg (average) across processes,
   * where * can be one of {t, f, f/s, m}.
   */
  static const ProfExpr GetProfField(const std::string& name);

  /**
   * Create a named profiling field from a given expr object.
   */
  static void SetProfField(const std::string& name, const ProfExpr& expr);

  /**
   * Construct a profiling expression from a given unary operator op acting on
   * the output of e1.
   */
  template <class UnaryOp> static ProfExpr UnaryExpr(const ProfExpr& e1, const UnaryOp& op);

  /**
   * Construct a profiling expression from a given binary operator op acting on
   * the output of e1 and e2.
   */
  template <class BinaryOp> static ProfExpr BinaryExpr(const ProfExpr& e1, const ProfExpr& e2, const BinaryOp& op);

  /**
   * Construct a profiling expression by applying a distributed reduction
   * operation on the output of e1.
   */
  static ProfExpr CommReduceExpr(const ProfExpr& e1, const CommOp comm_op);

  /**
   * Display the profiling output.
   *
   * @param[in] comm_ptr pointer to Comm object (can be nullptr).
   *
   * @param[in] fields list of fields to display in the profiling output.
   *
   * @param[in] format output format for each field.
   */
  static void print(const Comm* comm_ptr = nullptr, std::vector<std::string> fields = {}, std::vector<std::string> format = {});

  /**
   * Clear all profiling data.
   */
  static void reset();

 private:

  struct ProfileData;

  static ProfileData& GetProfData();


  class ExprScalar;

  template <class UnaryOp> class ExprUnary;

  template <class BinaryOp> class ExprBinary;

  friend ProfExpr operator+(const Profile::ProfExpr& u, const Profile::ProfExpr& v);

  friend ProfExpr operator-(const Profile::ProfExpr& u, const Profile::ProfExpr& v);

  friend ProfExpr operator*(const Profile::ProfExpr& u, const Profile::ProfExpr& v);

  friend ProfExpr operator/(const Profile::ProfExpr& u, const Profile::ProfExpr& v);

  friend ProfExpr operator*(const Profile::ProfExpr& u, const double a);

};

}  // end namespace

#endif // _SCTL_PROFILE_HPP_
