"""验证 HermesLite 包可以被正常导入。"""

import hermes_lite


def test_package_can_be_imported() -> None:
    """最小项目至少应能被测试环境导入。"""
    assert "HermesLite" in hermes_lite.__doc__