"""Stand-in for threadpoolctl on the machine without a GPU (edc_dmrg_gpu2 imports it; only DMRGGPU1t uses it)."""


class ThreadpoolController:
    def info(self):
        return []

    def limit(self, limits=None, user_api=None):
        class _Ctx:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass
        return _Ctx()
