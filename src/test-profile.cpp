// Per-function tests for sctl/profile.{hpp,txx}.
//
// Public API: Profile::Enable, Tic, Toc, IncrementCounter, Scoped,
// reset, print, ProfileCounter enum. The active implementation
// requires SCTL_PROFILE >= 0 (the Makefile defines SCTL_PROFILE=5).

#include <cstdio>
#include <string>

#include "sctl/common.hpp"
#include "sctl/profile.hpp"
#include "sctl/profile.txx"

#include "test-utils.hpp"

using sctl::Long;
using sctl::Profile;
using sctl::ProfileCounter;

int main() {
  // Start from a clean profiling state so previous-test residue doesn't leak in.
  Profile::reset();

  // --- Enable / disable: returns the previous state ---
  std::printf("Enable / disable :\n");
  {
    const bool was = Profile::Enable(false);
    CHECK(Profile::Enable(false) == false);   // already disabled
    CHECK(Profile::Enable(true)  == false);   // was disabled -> enabling, returns previous (false)
    CHECK(Profile::Enable(true)  == true);    // already enabled
    Profile::Enable(was);                     // restore
  }

  // --- IncrementCounter: returns OLD value via atomic fetch_add, accumulates ---
  // Counters are process-global singletons (no public API zeroes them), so we
  // verify accumulation via deltas rather than absolute values.
  std::printf("IncrementCounter :\n");
  {
    const Long c0 = Profile::IncrementCounter(ProfileCounter::CUSTOM1,  10);
    CHECK(Profile::IncrementCounter(ProfileCounter::CUSTOM1,   5) == c0 + 10);
    CHECK(Profile::IncrementCounter(ProfileCounter::CUSTOM1,  -3) == c0 + 15);
    CHECK(Profile::IncrementCounter(ProfileCounter::CUSTOM1,   0) == c0 + 12);

    // distinct counters don't bleed into each other
    const Long a0 = Profile::IncrementCounter(ProfileCounter::CUSTOM2, 0);
    const Long b0 = Profile::IncrementCounter(ProfileCounter::CUSTOM3, 0);
    Profile::IncrementCounter(ProfileCounter::CUSTOM2, 100);
    Profile::IncrementCounter(ProfileCounter::CUSTOM3, 200);
    CHECK(Profile::IncrementCounter(ProfileCounter::CUSTOM2, 0) == a0 + 100);
    CHECK(Profile::IncrementCounter(ProfileCounter::CUSTOM3, 0) == b0 + 200);
  }

  // --- reset() clears the Tic/Toc log without throwing ---
  // (Note: reset() does not zero the IncrementCounter atomics; it only clears
  // the timing/event log used by print().)
  std::printf("reset (log) :\n");
  {
    Profile::reset();
    CHECK(true);
  }

  // --- Tic / Toc: nested instrumentation runs without crashing ---
  // (We can't read back wall-time deltas from a public API, so this is a
  // smoke check that Tic/Toc pair correctly under several nesting levels.)
  std::printf("Tic / Toc nesting :\n");
  {
    Profile::reset();
    const bool prev = Profile::Enable(true);
    Profile::Tic("outer");
      Profile::Tic("inner1");
      Profile::Toc();
      Profile::Tic("inner2");
        Profile::Tic("innermost");
        Profile::Toc();
      Profile::Toc();
    Profile::Toc();
    Profile::Enable(prev);
    CHECK(true);  // reached here without segfault
  }

  // --- Scoped RAII: ctor=Tic, dtor=Toc, nesting works ---
  std::printf("Scoped RAII :\n");
  {
    const bool prev = Profile::Enable(true);
    const Long c0 = Profile::IncrementCounter(ProfileCounter::CUSTOM4, 0);
    {
      Profile::Scoped s_outer("scoped-outer");
      {
        Profile::Scoped s_inner("scoped-inner");
        Profile::IncrementCounter(ProfileCounter::CUSTOM4, 7);
      }
      Profile::IncrementCounter(ProfileCounter::CUSTOM4, 3);
    }
    Profile::Enable(prev);
    CHECK(Profile::IncrementCounter(ProfileCounter::CUSTOM4, 0) == c0 + 10);
  }

  // --- print() runs without crashing ---
  // (We don't parse the output; just confirm it doesn't throw / segfault.)
  std::printf("print smoke :\n");
  {
    Profile::print();
    CHECK(true);
  }

  // --- ProfileCounter enum order: TIME is the first field ---
  std::printf("ProfileCounter enum :\n");
  {
    CHECK((Long)ProfileCounter::TIME == 0);
    CHECK((Long)ProfileCounter::FIELD_COUNT > (Long)ProfileCounter::CUSTOM5);
  }

  TEST_SUMMARY_RETURN();
}
