"""Guard: classes shaped like embedding providers expose the protocol's public members.

``EmbeddingProviderProtocol`` is **structural**. A store accepts whatever is passed as
``embedding_provider=`` and only discovers a missing member when it reads one, so a provider that
keeps its dimension in a private attribute constructs cleanly and fails later, inside a retrieval
call — and only on the code paths that happen to read it, which differ between engrava versions.

This module scans the repository statically (``ast``) for classes that *look* like providers, and
requires them to declare the protocol's public members. The scan is static rather than import-based
because test modules define providers too, and importing every module to inspect one would pull in
fixtures, clients and engrava itself.

**What this guard does not do**, stated so its green is not read as more than it is:

- it recognises a provider by shape (``embed`` + ``embed_batch`` defined as methods), not by
  tracing what is actually handed to a store — an unrelated class of that shape is flagged, and a
  provider built dynamically (a factory, ``type()``, ``setattr``, ``__getattr__``) is not seen;
- ``REQUIRED_MEMBERS`` and ``PROVIDER_METHODS`` restate part of the protocol by hand, so a protocol
  change, a signature mismatch or a new required member goes unnoticed;
- inheritance resolves only within one module, so a class inheriting its members from an imported
  base is reported rather than passed;
- it recognises the common binding forms (a ``def``, an assignment, an annotated assignment with a
  value) and not the unusual ones (tuple unpacking, augmented assignment, a binding made inside
  class-body control flow, an attribute set on ``self``);
- it checks that a name is bound in the source, not that it resolves at runtime.

It catches the one mistake it was written for — a private ``_dimension`` with no public property —
cheaply and offline. Its green means that mistake is absent from the shapes it can see; treat
everything above as unguarded.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Directories that can define an embedding provider.
SCANNED_DIRS = ("adapters", "integrations", "runners", "tests")

# The two members `EmbeddingProviderProtocol` requires as PUBLIC attributes.
REQUIRED_MEMBERS = ("dimension", "model_name")

# A class is an embedding provider, structurally, if it defines these.
PROVIDER_METHODS = frozenset({"embed", "embed_batch"})


def _public_members(node: ast.ClassDef) -> set[str]:
    """Collect the public names a class body binds.

    Counts methods, properties and class-level assignments alike: the protocol only cares that
    ``provider.dimension`` resolves, not how.

    Args:
        node: The class definition to inspect.

    Returns:
        The set of public attribute names the class body binds directly.
    """
    names: set[str] = set()
    for item in node.body:
        if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
            names.add(item.name)
        elif (
            isinstance(item, ast.AnnAssign)
            and item.value is not None
            and isinstance(item.target, ast.Name)
        ):
            # An annotation with no value binds nothing at runtime: ``dimension: int`` alone
            # would still raise AttributeError on read, so it must not count as satisfying
            # the protocol.
            names.add(item.target.id)
        elif isinstance(item, ast.Assign):
            names.update(t.id for t in item.targets if isinstance(t, ast.Name))
    return {n for n in names if not n.startswith("_")}


def _is_protocol_base(base: ast.expr) -> bool:
    """Return whether a base expression names ``typing.Protocol``.

    Matches the bare ``Protocol``, the qualified ``typing.Protocol``, and either subscripted
    (``Protocol[T]``). A Protocol declaration is the contract rather than an implementation of it,
    so it is not held to the requirement. Deliberately narrow: an unrelated ``something.Protocol``
    is NOT excluded, because excluding it would silently drop a real class from the scan.

    Args:
        base: One entry from a class definition's base list.

    Returns:
        Whether the base names ``typing.Protocol``.
    """
    if isinstance(base, ast.Subscript):
        base = base.value
    if isinstance(base, ast.Name):
        return base.id == "Protocol"
    return (
        isinstance(base, ast.Attribute)
        and base.attr == "Protocol"
        and isinstance(base.value, ast.Name)
        and base.value.id == "typing"
    )


Scope = tuple[str, ...]


def _module_classes(tree: ast.Module) -> dict[Scope, ast.ClassDef]:
    """Index a module's class definitions by their lexical scope.

    Keying by name alone would let two classes that share a name in different scopes collide, so
    one would be discarded and a base name could resolve against the wrong body. The key is the
    enclosing path instead, which keeps both and lets resolution prefer the nearest scope.

    Args:
        tree: The parsed module.

    Returns:
        Class definitions by scope path, e.g. ``("f", "P")`` for a ``P`` nested in ``def f``.
    """
    classes: dict[Scope, ast.ClassDef] = {}

    def walk(node: ast.AST, scope: Scope) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                classes[(*scope, child.name)] = child
                walk(child, (*scope, child.name))
            elif isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                walk(child, (*scope, child.name))
            else:
                walk(child, scope)

    walk(tree, ())
    return classes


def _resolve(name: str, scope: Scope, classes: dict[Scope, ast.ClassDef]) -> Scope | None:
    """Find the class a bare base name refers to, nearest enclosing scope first.

    Args:
        name: The base class name as written.
        scope: The scope path of the class whose bases are being resolved.
        classes: The module's class index.

    Returns:
        The resolved scope path, or ``None`` when the name is not defined in this module.
    """
    for depth in range(len(scope), -1, -1):
        candidate = (*scope[:depth], name)
        if candidate in classes:
            return candidate
    return None


def _inherited_members(scope: Scope, classes: dict[Scope, ast.ClassDef]) -> set[str]:
    """Resolve a class's public members, including those inherited within the module.

    A subclass satisfies the protocol through its base, so requiring it to redeclare a member
    would make this guard demand duplication. Only bases defined in this module resolve; a base
    imported from elsewhere is invisible to a static read, so such a class is reported rather than
    silently passed.

    Args:
        scope: The scope path of the class to resolve.
        classes: The module's class index.

    Returns:
        The public members visible on the class, own and inherited.
    """
    seen: set[Scope] = set()
    members: set[str] = set()
    stack = [scope]
    while stack:
        current = stack.pop()
        if current in seen or current not in classes:
            continue
        seen.add(current)
        node = classes[current]
        members |= _public_members(node)
        for base in node.bases:
            if isinstance(base, ast.Name):
                resolved = _resolve(base.id, current[:-1], classes)
                if resolved is not None:
                    stack.append(resolved)
    return members


def _defines_provider_methods(scope: Scope, classes: dict[Scope, ast.ClassDef]) -> bool:
    """Return whether a class defines the protocol's distinguishing methods.

    Checks for actual ``def``s, own or inherited: a class attribute merely *named* ``embed`` is not
    a provider, and treating it as one would report a class that could never be passed to a store.

    Args:
        scope: The scope path of the class to check.
        classes: The module's class index.

    Returns:
        Whether both distinguishing methods are defined.
    """
    seen: set[Scope] = set()
    methods: set[str] = set()
    stack = [scope]
    while stack:
        current = stack.pop()
        if current in seen or current not in classes:
            continue
        seen.add(current)
        node = classes[current]
        methods |= {
            item.name
            for item in node.body
            if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef)
        }
        for base in node.bases:
            if isinstance(base, ast.Name):
                resolved = _resolve(base.id, current[:-1], classes)
                if resolved is not None:
                    stack.append(resolved)
    return methods >= PROVIDER_METHODS


def _discover_providers() -> list[tuple[str, str, set[str]]]:
    """Find the classes in the scanned directories that are shaped like providers.

    Returns:
        One ``(module_path, dotted_class_path, public_members)`` triple per match, sorted for a
        stable parameter order.
    """
    found: list[tuple[str, str, set[str]]] = []
    for directory in SCANNED_DIRS:
        for path in sorted((REPO_ROOT / directory).rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            classes = _module_classes(tree)
            for scope, node in classes.items():
                if any(_is_protocol_base(base) for base in node.bases):
                    continue
                if not _defines_provider_methods(scope, classes):
                    continue
                rel = path.relative_to(REPO_ROOT).as_posix()
                found.append((rel, ".".join(scope), _inherited_members(scope, classes)))
    return sorted(found, key=lambda item: (item[0], item[1]))


PROVIDERS = _discover_providers()


def test_discovery_is_not_vacuous() -> None:
    """The scan must actually find providers.

    A rename or a moved directory would otherwise reduce this module to zero parametrised cases
    and still report green. This catches only the literal zero case; it cannot tell a real provider
    from an unrelated class of the same shape.
    """
    assert PROVIDERS, f"no embedding providers discovered under {SCANNED_DIRS}"
    scanned_dirs = {path.split("/", 1)[0] for path, _, _ in PROVIDERS}
    assert {"integrations", "tests"} <= scanned_dirs, (
        f"expected providers in integrations/ and tests/, found only {sorted(scanned_dirs)}"
    )


@pytest.mark.parametrize(
    ("module_path", "class_name", "members"),
    PROVIDERS,
    ids=[f"{path}::{name}" for path, name, _ in PROVIDERS],
)
def test_provider_exposes_required_public_members(
    module_path: str, class_name: str, members: set[str]
) -> None:
    """A class shaped like an embedding provider exposes the protocol's public members.

    Args:
        module_path: Repo-relative path of the defining module.
        class_name: The provider class.
        members: Public names bound in the class body.
    """
    missing = [m for m in REQUIRED_MEMBERS if m not in members]
    assert not missing, (
        f"{module_path}::{class_name} has the shape of an embedding provider but does not expose "
        f"{missing}. EmbeddingProviderProtocol is structural, so such a class is accepted at "
        f"construction and fails inside a retrieval call instead. A private attribute such as "
        f"'_{missing[0]}' does not satisfy it — add a public property."
    )


# --- the analyzer's own behaviour ------------------------------------------------------------
#
# The scan above only ever runs against this repository, so its edge cases would otherwise be
# exercised by whatever the repo happens to contain today. These cases pin the behaviour directly.


def _members_of(source: str, *scope: str) -> set[str]:
    """Resolve a class's visible public members from source text.

    Args:
        source: A module's source.
        scope: The class's scope path, e.g. ``("f", "P")`` for a ``P`` nested in ``def f``.

    Returns:
        The public members the analyzer sees.
    """
    return _inherited_members(scope, _module_classes(ast.parse(source)))


def test_annotation_without_value_does_not_count_as_a_member() -> None:
    """``dimension: int`` binds nothing at runtime, so it must not satisfy the requirement."""
    source = "class P:\n    dimension: int\n    model_name = 'm'\n"
    assert _members_of(source, "P") == {"model_name"}


def test_annotated_assignment_with_a_value_counts() -> None:
    """``dimension: int = 8`` does bind, so it satisfies the requirement."""
    source = "class P:\n    dimension: int = 8\n"
    assert "dimension" in _members_of(source, "P")


def test_members_are_inherited_within_a_module() -> None:
    """A subclass need not redeclare what its base already exposes."""
    source = "class Base:\n    dimension = 8\n\n\nclass Sub(Base):\n    model_name = 'm'\n"
    assert {"dimension", "model_name"} <= _members_of(source, "Sub")


def test_private_names_are_not_members() -> None:
    """A private attribute does not satisfy a protocol that requires a public one."""
    source = "class P:\n    _dimension = 8\n"
    assert _members_of(source, "P") == set()


def test_same_named_classes_in_different_scopes_are_both_kept() -> None:
    """A name collision must keep both classes, not discard one."""
    source = "class P:\n    dimension = 1\n\n\ndef f():\n    class P:\n        other = 2\n"
    assert _members_of(source, "P") == {"dimension"}
    assert _members_of(source, "f", "P") == {"other"}


def test_inheritance_resolves_in_the_nearest_scope() -> None:
    """A nested base shadows a top-level one of the same name."""
    source = (
        "class Base:\n    dimension = 1\n\n\n"
        "def f():\n"
        "    class Base:\n        model_name = 'inner'\n\n"
        "    class Sub(Base):\n        pass\n"
    )
    assert _members_of(source, "f", "Sub") == {"model_name"}


def test_a_non_callable_named_embed_is_not_a_provider() -> None:
    """Provider shape means methods; an attribute merely named ``embed`` is not one."""
    source = "class P:\n    embed = 1\n    embed_batch = 2\n"
    classes = _module_classes(ast.parse(source))
    assert not _defines_provider_methods(("P",), classes)


def test_provider_methods_may_be_inherited() -> None:
    """A subclass of a provider is a provider."""
    source = (
        "class Base:\n"
        "    async def embed(self): ...\n"
        "    async def embed_batch(self): ...\n\n\n"
        "class Sub(Base):\n    pass\n"
    )
    classes = _module_classes(ast.parse(source))
    assert _defines_provider_methods(("Sub",), classes)


@pytest.mark.parametrize(
    "base",
    ["Protocol", "typing.Protocol", "Protocol[int]"],
    ids=["bare", "qualified", "subscripted"],
)
def test_protocol_definitions_are_recognised(base: str) -> None:
    """A Protocol declaration is the contract, not an implementation of it."""
    node = ast.parse(f"class P({base}):\n    pass\n").body[0]
    assert isinstance(node, ast.ClassDef)
    assert _is_protocol_base(node.bases[0])


def test_a_plain_base_is_not_mistaken_for_a_protocol() -> None:
    """Only ``Protocol`` is excluded; an ordinary base class is still an implementation."""
    node = ast.parse("class P(Base):\n    pass\n").body[0]
    assert isinstance(node, ast.ClassDef)
    assert not _is_protocol_base(node.bases[0])


def test_discovery_excludes_protocol_declarations_but_not_implementations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The exclusion is applied by discovery, not only available to it.

    Testing :func:`_is_protocol_base` alone would stay green if discovery stopped calling it.
    """
    module = tmp_path / "sample" / "mod.py"
    module.parent.mkdir()
    module.write_text(
        "from typing import Protocol\n\n\n"
        "class Contract(Protocol):\n"
        "    async def embed(self): ...\n"
        "    async def embed_batch(self): ...\n\n\n"
        "class Impl:\n"
        "    dimension = 8\n"
        "    model_name = 'm'\n"
        "    async def embed(self): ...\n"
        "    async def embed_batch(self): ...\n",
        encoding="utf-8",
    )
    # Patch this module's own globals. A dotted-path setattr would resolve a second import of
    # this file under a different module name and leave the copy under test untouched.
    monkeypatch.setitem(globals(), "REPO_ROOT", tmp_path)
    monkeypatch.setitem(globals(), "SCANNED_DIRS", ("sample",))

    names = {name for _, name, _ in _discover_providers()}

    assert names == {"Impl"}, f"expected only the implementation, got {sorted(names)}"
