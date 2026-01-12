import Lake
open Lake DSL

package laaciq where
  -- Configuration for the LA-ACIQ formalization

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.3.0"

@[default_target]
lean_lib Laaciq where
  roots := #[`Laaciq]
