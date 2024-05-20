#ifndef _SCTL_MEM_MGR_HPP_
#define _SCTL_MEM_MGR_HPP_

#include <map>              // for __map_iterator, multimap
#include <set>              // for set
#include <stack>            // for stack
#include <typeinfo>         // for type_info
#include <vector>           // for vector

#include "sctl/common.hpp"  // for Long, SCTL_NAMESPACE

// TODO: Implement fast stack allocation.

namespace SCTL_NAMESPACE {

/**
 * \brief MemoryManager class declaration.
 */
class MemoryManager {

 public:
  static constexpr char init_mem_val = 42;
  static constexpr Long end_padding = 64;

  /**
   * \brief Header data for each memory block.
   */
  struct MemHead {
    typedef decltype(typeid(char).hash_code()) TypeID;
    Long n_indx;
    Long n_elem;
    Long type_size;
    Long alloc_ctr;
    TypeID type_id;
    #ifdef SCTL_MEMDEBUG
    unsigned char check_sum;
    #endif
  };

  /**
   * \brief Constructor for MemoryManager.
   */
  explicit MemoryManager(Long N);

  /**
   * \brief Constructor for MemoryManager.
   */
  ~MemoryManager();

  static MemHead& GetMemHead(char* p);

  static void CheckMemHead(const MemHead& p);

  Iterator<char> malloc(const Long n_elem, const Long type_size = sizeof(char), const MemHead::TypeID type_id = typeid(char).hash_code()) const;

  void free(Iterator<char> p) const;

  void print() const;

  static void test();

  // Check all free memory equals init_mem_val
  void Check() const;

  // A global MemoryManager object. This is the default for aligned_new and aligned_free
  static MemoryManager& glbMemMgr() {
    static MemoryManager m(SCTL_GLOBAL_MEM_BUFF * 1024LL * 1024LL);
    return m;
  }

 private:
  // Private constructor
  MemoryManager();

  // Private copy constructor
  MemoryManager(const MemoryManager& m);

  /**
   * \brief Node structure for a doubly linked list, representing free and
   * occupied memory blocks. Blocks are split, merged or state is changed
   * between free and occupied in O(1) time given the pointer to the MemNode.
   */
  struct MemNode {
    bool free;
    Long size;
    char* mem_ptr;
    Long prev, next;
    std::multimap<Long, Long>::iterator it;
  };

  /**
   * \brief Return index of one of the available MemNodes from node_stack or
   * create new MemNode by resizing node_buff.
   */
  Long new_node() const;

  /**
   * \brief Add node index for now available MemNode to node_stack.
   */
  void delete_node(Long indx) const;

  char* buff;         // pointer to memory buffer.
  Long buff_size;     // total buffer size in bytes.
  Long n_dummy_indx;  // index of first (dummy) MemNode in link list.

  mutable std::vector<MemNode> node_buff;      // storage for MemNode objects, this can only grow.
  mutable std::stack<Long> node_stack;         // stack of available free MemNodes from node_buff.
  mutable std::multimap<Long, Long> free_map;  // pair (MemNode.size, MemNode_id) for all free MemNodes.
  //mutable omp_lock_t omp_lock;                 // openmp lock to prevent concurrent changes.
  //mutable std::mutex mutex_lock;
  mutable std::set<void*> system_malloc;       // track pointers allocated using system malloc.
};

/**
 * \brief Aligned allocation as an alternative to new. Uses placement new to
 * construct objects.
 */
template <class ValueType> Iterator<ValueType> aligned_new(Long n_elem = 1, const MemoryManager* mem_mgr = &MemoryManager::glbMemMgr());

/**
 * \brief Aligned de-allocation as an alternative to delete. Calls the object
 * destructor.
 */
template <class ValueType> void aligned_delete(Iterator<ValueType> A, const MemoryManager* mem_mgr = &MemoryManager::glbMemMgr());

}  // end namespace SCTL_NAMESPACE

#endif // _SCTL_MEM_MGR_HPP_
