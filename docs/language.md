# Federated Language

[TOC]

## Federated Computation

```python
federated_language.framework.set_default_context(...)  # 3

@federated_language.federated_computation(np.int32)  # 2
def foo(x):  # 1
  return (x, x)

result = foo(2) # 4
```

1.  Write a *Python* function.

1.  Decorate the *Python* function using
    `@federated_language.federated_computation`.

    When the Python is parsed, the `@federated_language.federated_computation`
    decorator will [trace](#tracing) the Python function and create a
    `federated_language.Computation`.

1.  Install a [context](#context).

1.  Invoke the `federated_language.Computation`.

    When the `federated_language.Computation` is invoked, it is
    [compiled](#compilation) and [executed](#execution) in the
    [context](#context) that was installed.

## AST

The Abstract Syntax Tree (AST) in Federated Language defines the structure and
type signature of a `federated_language.Computation`.

### Computation

A `federated_language.Computation` is an abstraction defining the API for a
computation.

### ComputationBuildingBlock

A `federated_language.framework.ComputationBuildingBlock` is the Python
representation of the [AST](#ast).

### computation_pb2.Computation

A `computation_pb2.Computation` is the Proto or serialized representation of the
[AST](#ast).

## Tracing

Tracing the the process of creating a `federated_language.Computation` from a
Python function.

There are three steps to tracing a `federated_language.Computation`:

### Packing the Arguments

Internally, a `federated_language.Computation` can only ever have zero or one
argument. The arguments provided to `@federated_language.federated_computation`
describe type signature of the parameters to the
`federated_language.Computation`. This information is used to pack the arguments
of the Python function into a single object.

### Tracing the Function

When tracing a `@federated_language.federated_computation`, the Python function
is called using `federated_language.Value` as a substitute for each argument.
`federated_language.Value` emulates the behavior of the argument type by
implementing common Python dunder methods (e.g., `__getattr__`).

### Creating the Computation

The result of tracing the function is packaged into a
`federated_language.framework.Lambda` whose `parameter_name` and
`parameter_type` map to the `federated_language.framework.Reference` created to
represent the packed arguments. The resulting
`federated_language.framework.Lambda` is returned as a
`federated_language.Computation` that represents the Python function.

## Compilation

Compilation is the process of transforming an [AST](#ast) into a different
[AST](#ast) in order to provide some guarantee, optimization, property, etc.

A **compiler** is a sequence of **transformations**.

### Transformation

A **transformations** creates a new [AST](#ast) an existing one. Transformations
can operate on `federated_language.framework.ComputationBuildingBlock`s in order
to transform the Python representation of an [AST](#ast) or on
`computation_pb2.Computation`s in order to transform the Proto or serialized
representation of an [AST](#ast).

An **atomic transformation** is one that applies a single mutation (possibly
more than once) to the input.

A **composite transformation** is one that applies multiple transformations to
the input.

Transformations can be applied in serial or parallel, meaning that you can
create a composite transformation that performs multiple transformations in one
pass through an [AST](#ast). However, the order in which you apply
transformations and how those transformations are parallelized is hard to reason
about; as a result, composite transformations are hand-crafted and often
fragile.

## Execution

Execution is the process of executing a `federated_language.Computation`.

A **backend** or **runtime** is a logical construct describing a system that
executes a `federated_language.Computation`.

### Executor

An `federated_language.framework.Executor` is an abstract interface that defines
the API for executing a `federated_language.Computation`.

### ExecutorFactory

An `federated_language.framework.ExecutorFactory` is an abstract interface that
defines the API for creating an `federated_language.framework.Executor`. These
factories create the executor lazily and manage the lifecycle of the executor.
The motivation to lazily creating executors is to infer the number of clients at
execution time.

### Execution Stack

An **execution stack** is a logical construct describing a hierarchy of
`federated_language.framework.Executor`s.

## Context

A `federated_language.framework.SyncContext` or
`federated_language.framework.AsyncContext` are data structures that hold the
objects required to [trace](#tracing), [compile](#compilation), or
[execute](#execution) a `federated_language.Computation`.

### ContextStack

A `federated_language.framework.ContextStack` is a data structure that contains
a stack of [contexts](#context).

### FederatedComputationContext

A `federated_language.framework.FederatedComputationContext` is a
[context](#context) that [traces](#tracing) Python functions decorated with the
`@federated_language.federated_computation` decorator.

### ExecutionContext

An `federated_language.framework.ExecutionContext` is a [context](#context) that
[compiles](#compilation) `federated_language.Computation`s using a
**transformation** and [executes](#execution) `federated_language.Computation`s
using an `federated_language.framework.Executor`.

### Default Context

The **default context** is the logical construct describing the context at the
top of the `federated_language.framework.ContextStack` that is used to
[compile](#compilation) and [execute](#execution)
`federated_language.Computation`s.

You can set the **default context** by invoking
`federated_language.framework.set_default_context`.

## Platform

A platform is a [context](#context) that is used to execute
`federated_language.Computation`s and can be implemented using a variety of
technologies or systems.

Often a platform is implemented as an [ExecutionContext](#executioncontext)
using a [compiler](#compiler) and an [executor](#executor).
