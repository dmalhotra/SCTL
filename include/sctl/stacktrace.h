#ifndef _SCTL_STACKTRACE_H_
#define _SCTL_STACKTRACE_H_

#include <unistd.h>
#include <signal.h>         // for sigaction, SIGABRT, sigemptyset, siginfo_t
#include <stdio.h>          // for fprintf, stderr, fgets, FILE, pclose, popen
#include <stdlib.h>         // for exit
#include <execinfo.h>       // for backtrace, backtrace_symbols
#include <cxxabi.h>
#include <dlfcn.h>          // for dladdr, Dl_info — needed to translate
                            // runtime VMAs to file-relative offsets for PIE
                            // executables and shared libraries.
#ifdef __APPLE__
#include <mach-o/dyld.h>    // for _NSGetExecutablePath
#include <cstdint>          // for uint32_t
#endif

#include "sctl/common.hpp"  // for SCTL_UNUSED, sctl

namespace sctl {

inline void print_stacktrace(FILE* out = stderr, int skip = 1) {
  // Get addresses
  void* addrlist[256];
  int addrlen = backtrace(addrlist, 255);
  // Step back one byte so we point inside the call instruction, not at
  // the return address (which may be past the end of a [[noreturn]] callee).
  for (int i = 0; i < addrlen; i++) addrlist[i] = (char*)addrlist[i] - 1;

  // Get symbols
  char** symbollist = backtrace_symbols(addrlist, addrlen);

  // Fallback module path (/proc/self/exe on Linux, _NSGetExecutablePath on
  // macOS) used when dladdr can't identify the module.
  char fallback_exe[10240];
  fallback_exe[0] = '\0';
#ifdef __APPLE__
  uint32_t size = sizeof(fallback_exe);
  _NSGetExecutablePath(fallback_exe, &size);
#elif __linux__
  ssize_t fname_len = ::readlink("/proc/self/exe", fallback_exe, sizeof(fallback_exe) - 1);
  if (fname_len > 0) fallback_exe[fname_len] = '\0';
#endif

  // Run `cmd` and read up to two whitespace-stripped lines into buf0/buf1.
  // Returns true if buf0 looks like a real (non-error) symbolizer result.
  // `two_lines_out` reports whether both lines were captured (the typical
  // `addr2line -f` output: "function\nfile:line").
  auto run_query = [](const char* cmd, char* buf0, size_t buf_sz,
                      char* buf1, bool& two_lines_out) -> bool {
    two_lines_out = false;
    FILE* pipe = popen(cmd, "r");
    if (!pipe) return false;
    buf0[0] = '\0';
    buf1[0] = '\0';
    char* r0 = fgets(buf0, (int)buf_sz - 1, pipe);
    char* r1 = fgets(buf1, (int)buf_sz - 1, pipe);
    pclose(pipe);
    for (size_t j = 0; j < buf_sz - 1; j++) if (buf0[j] == '\n') buf0[j] = ' ';
    for (size_t j = 0; j < buf_sz - 1; j++) if (buf1[j] == '\n') buf1[j] = ' ';
    two_lines_out = (r0 != nullptr && r1 != nullptr);
    // addr2line returns "??" (function) and "??:0" (file:line) on miss.
    return r0 != nullptr && buf0[0] != '?' && buf0[0] != '\0' && buf0[0] != '0';
  };

  // Print
  for (int i = skip; i < addrlen; i++) {
    // Identify the module this frame belongs to. For PIE executables and
    // shared libraries the file VMA differs from the runtime VMA — we
    // need to pass (runtime - load_base) to addr2line, not the runtime
    // address. dladdr also gives us the right .so path for frames in
    // shared libraries (libc, etc.) instead of forcing the main exe.
    Dl_info dl_info{};
    const char* mod_path = nullptr;
    [[maybe_unused]] void* mod_offset = addrlist[i];
    if (dladdr(addrlist[i], &dl_info) && dl_info.dli_fname) {
      mod_path = dl_info.dli_fname;
      mod_offset = (void*)((char*)addrlist[i] - (char*)dl_info.dli_fbase);
    }
    if (!mod_path || !*mod_path) mod_path = fallback_exe;

    char buffer0[10240];
    char buffer1[10240];
    bool two_lines = false;
    bool ok = false;
    if (mod_path && *mod_path) {
      char cmd[sizeof(fallback_exe) + 64];
#ifdef __APPLE__
      // atos accepts the runtime address with -l <load_base>, or the file
      // VMA without. With -Wl,-no_pie (set in the Makefile for macOS) the
      // binary VMA equals the runtime address, so we pass that directly.
      snprintf(cmd, sizeof(cmd), "atos -o %s -l %p %p 2> /dev/null",
               mod_path, dl_info.dli_fbase, addrlist[i]);
#elif __linux__
      // First try the file-relative offset (correct for PIE / shared libs).
      snprintf(cmd, sizeof(cmd), "addr2line -f -C -i -e %s %p 2> /dev/null",
               mod_path, mod_offset);
#endif
      ok = run_query(cmd, buffer0, sizeof(buffer0), buffer1, two_lines);
#ifdef __linux__
      // Fallback: non-PIE binaries (ET_EXEC) want the raw runtime VMA.
      if (!ok && mod_offset != addrlist[i]) {
        snprintf(cmd, sizeof(cmd), "addr2line -f -C -i -e %s %p 2> /dev/null",
                 mod_path, addrlist[i]);
        ok = run_query(cmd, buffer0, sizeof(buffer0), buffer1, two_lines);
      }
#endif
    }

    if (ok && !two_lines) {
      fprintf(out, "[%d] %s\n", i - skip, buffer0);
    } else if (ok && two_lines) {
      fprintf(out, "[%d] %s: %s\n", i - skip, buffer1, buffer0);
    } else {
      // Fallback: demangle the symbol from backtrace_symbols if possible.
      // glibc format: "module(_Zmangled+0xoffset) [0xaddr]"
      // macOS  format: "idx module 0xaddr _Zmangled + offset"
      char* sym = symbollist[i];
      char* mangled_begin = nullptr;
      char* mangled_end = nullptr;
      for (char* p = sym; *p; p++) {
        if (*p == '(' && !mangled_begin) { mangled_begin = p + 1; }
        else if ((*p == '+' || *p == ')') && mangled_begin && !mangled_end) { mangled_end = p; break; }
      }
      char* demangled = nullptr;
      if (mangled_begin && mangled_end && mangled_end > mangled_begin) {
        char saved = *mangled_end;
        *mangled_end = '\0';
        int status = 0;
        demangled = abi::__cxa_demangle(mangled_begin, nullptr, nullptr, &status);
        *mangled_end = saved;
        if (status != 0) { free(demangled); demangled = nullptr; }
      }
      if (demangled) {
        char saved = *mangled_begin;
        *mangled_begin = '\0';
        fprintf(out, "[%d] %p: %s%s%s\n", i - skip, addrlist[i], sym, demangled, mangled_end);
        *mangled_begin = saved;
        free(demangled);
      } else {
        fprintf(out, "[%d] %p: %s\n", i - skip, addrlist[i], sym);
      }
    }
  }
  free(symbollist);
  fprintf(stderr, "\n");
}

inline void abortHandler(int signum, siginfo_t* si, void* unused) {
  static bool first_time = true;
  SCTL_UNUSED(unused);
  SCTL_UNUSED(si);

#pragma omp critical(SCTL_STACK_TRACE)
  if (first_time) {
    first_time = false;
    const char* name = nullptr;
    switch (signum) {
      case SIGABRT:
        name = "SIGABRT";
        break;
      case SIGSEGV:
        name = "SIGSEGV";
        break;
      case SIGBUS:
        name = "SIGBUS";
        break;
      case SIGILL:
        name = "SIGILL";
        break;
      case SIGFPE:
        name = "SIGFPE";
        break;
    }

    if (name)
      fprintf(stderr, "\nCaught signal %d (%s)\n", signum, name);
    else
      fprintf(stderr, "\nCaught signal %d\n", signum);

    print_stacktrace(stderr, 2);
  }
  exit(signum);
}

inline int SetSigHandler() {
  struct sigaction sa;
  sa.sa_flags = SA_RESTART | SA_SIGINFO;
  sa.sa_sigaction = abortHandler;
  sigemptyset(&sa.sa_mask);

  sigaction(SIGABRT, &sa, nullptr);
  sigaction(SIGSEGV, &sa, nullptr);
  sigaction(SIGBUS, &sa, nullptr);
  sigaction(SIGILL, &sa, nullptr);
  sigaction(SIGFPE, &sa, nullptr);
  sigaction(SIGPIPE, &sa, nullptr);

  return 0;
}


#ifdef SCTL_SIG_HANDLER
inline int sgh = sctl::SetSigHandler(); // Set signal handler
#endif

}  // end namespace

#endif // _SCTL_STACKTRACE_H_
