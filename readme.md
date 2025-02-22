Here's a GitHub README.md for your project:

# Rust Zanzibar Authorization

A Rust implementation of Google's Zanzibar authorization system with extended functionality. This implementation is inspired by the [Google Zanzibar paper](https://research.google/pubs/pub48190/) but built from scratch with additional features.

## Overview

This project implements the Zanzibar consistent authorization system in Rust, providing a flexible and scalable solution for managing permissions across distributed systems. While following the core concepts of Zanzibar, it extends the functionality with additional relationship models.

## Key Features

- Core Zanzibar relationship-based authorization
- Extended relationship models
- High-performance Rust implementation
- Custom namespace configurations
- Flexible permission checking

## Extended Models

### Document Hierarchy Example

The following shape demonstrates a document hierarchy with inherited permissions:

```yaml
entity document {
    relation org @organization
    relation viewer @user @group#direct_member @group#manager
    relation manager @user @group#direct_member @group#manager

    action edit = manager or org.admin
    action view = viewer or manager or org.admin
}
```

This model allows:
- Direct user assignments as viewers/managers
- Group-based access through membership
- Organizational admin override
- Inherited permissions through the hierarchy

### Permission Check Example

To check if a user has permission to edit a document:

```rust
// Check if user:alice can edit document:report1
let check = {
    entity: "document:report1",
    subject: "user:alice",
    permission: "edit"
};

// Returns true if:
// 1. Alice is a direct manager of the document
// 2. Alice is in a group that has manager access
// 3. Alice is an admin of the organization that owns the document
```

## Implementation Details

- Built in Rust for high performance and memory safety
- Uses a graph-based relationship model
- Supports concurrent permission checks
- Efficient caching mechanisms
- Configurable consistency levels

## Getting Started

1. Add to your Cargo.toml:
```toml
[dependencies]
zanzibar = "0.1.0"
```

2. Define your authorization model in YAML
3. Initialize the authorization system
4. Start checking permissions

## How to use
> cargo test -- --nocapture

```
running 1 test
Running Zanzibar permission tests:

banking              ✅ PASSED
disney_plus          ✅ PASSED
mercury              ✅ PASSED
organisation         ✅ PASSED
rbac                 ✅ PASSED
user_groups          ✅ PASSED
custom_roles         ✅ PASSED
google_docs          ✅ PASSED
instagram            ✅ PASSED
document_management  ✅ PASSED
hospital_management  ✅ PASSED
content_moderation   ✅ PASSED
document_hierarchy   ✅ PASSED

Test Summary:
  Passed: 13
  Failed: 0
  Total:  13
test tests::test_zanzibar_permissions ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.22s
```

> ZANZIBAR_DEBUG=true cargo test -- --nocapture

Will give additional debug information

## License

This project is licensed under the MIT License.

## Disclaimer

This is an independent implementation inspired by the Google Zanzibar paper. No code has been used from other implementations like Permify - only the structure of the shapes has been used as reference.

---

Note: This implementation is focused on providing a robust, production-ready authorization system while maintaining the flexibility to extend beyond the original Zanzibar model.