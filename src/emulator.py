"""RISC-V emulator from https://fmash16.github.io/content/posts/riscv-emulator-in-c.html"""

import logging
from typing import List

logging.basicConfig()
logger = logging.getLogger()


class Ram:
    SIZE: int = 1024 * 1024 + 1
    BASE: int = 0x80000000

    def __init__(self) -> None:
        self._raw = bytearray(self.SIZE)

    def load(self, addr: int, size: int) -> int:
        result = 0
        for i in range(size // 8):
            result |= self._raw[addr - self.BASE + i] << (i * 8)
        return result

    def store(self, addr: int, size: int, value: int) -> None:
        for i in range(size // 8):
            self._raw[addr - self.BASE + i] = (value >> (i * 8)) & 0xFF

    def read_file(self, path: str) -> None:
        with open(path, "rb") as binfile:
            data = binfile.read(self.SIZE + 1)

            assert len(data) <= self.SIZE, "Path is too large!"
            self._raw = bytearray(data)


class Bus:
    def __init__(self, ram: Ram):
        self._ram = ram

    def load(self, addr: int, size: int) -> int:
        return self._ram.load(addr, size)

    def store(self, addr: int, size: int, value: int) -> None:
        self._ram.store(addr, size, value)


class Cpu:
    # Opcodes for different instruction types
    OP_LOAD = 0b0000011
    OP_IMM = 0b0010011
    OP_STORE = 0b0100011
    OP_BRANCH = 0b1100011
    OP_JAL = 0b1101111
    OP_JALR = 0b1100111
    OP_LUI = 0b0110111
    OP_AUIPC = 0b0010111
    OP_SYSTEM = 0b1110011

    # Funct3 values for immediate-type instructions
    FUNCT3_ADDI = 0b000
    FUNCT3_SLTI = 0b010
    FUNCT3_SLTIU = 0b011
    FUNCT3_XORI = 0b100
    FUNCT3_ORI = 0b110
    FUNCT3_ANDI = 0b111

    # Funct7 values for R-type instructions
    FUNCT7_ADD_SUB = 0b0000000
    FUNCT7_SRL_SRA = 0b0000000
    FUNCT7_MUL_DIV = 0b0000001
    FUNCT7_SLLI = 0b0000000
    FUNCT7_SRLI = 0b0000000
    FUNCT7_SRAI = 0b0100000  # SRAI differs in FUNCT7 value from SRLI

    # Funct3 values for R-type instructions
    FUNCT3_ADD_SUB = 0b000
    FUNCT3_SLL = 0b001
    FUNCT3_SLT = 0b010
    FUNCT3_SLTU = 0b011
    FUNCT3_XOR = 0b100
    FUNCT3_SRL_SRA = 0b101
    FUNCT3_OR = 0b110
    FUNCT3_AND = 0b111

    # Funct12 values for SYSTEM instructions
    FUNCT12_ECALL = 0b000000000000
    FUNCT12_EBREAK = 0b000000000001

    # Register names (x0 to x31)
    REGISTER_ZERO = 0
    REGISTER_RA = 1
    REGISTER_SP = 2
    REGISTER_GP = 3
    REGISTER_TP = 4
    REGISTER_T0 = 5
    REGISTER_T1 = 6
    REGISTER_T2 = 7
    REGISTER_S0 = 8
    REGISTER_FP = 8
    REGISTER_S1 = 9
    REGISTER_A0 = 10
    REGISTER_A1 = 11
    REGISTER_A2 = 12
    REGISTER_A3 = 13
    REGISTER_A4 = 14
    REGISTER_A5 = 15
    REGISTER_A6 = 16
    REGISTER_A7 = 17
    REGISTER_S2 = 18
    REGISTER_S3 = 19
    REGISTER_S4 = 20
    REGISTER_S5 = 21
    REGISTER_S6 = 22
    REGISTER_S7 = 23
    REGISTER_S8 = 24
    REGISTER_S9 = 25
    REGISTER_S10 = 26
    REGISTER_S11 = 27
    REGISTER_T3 = 28
    REGISTER_T4 = 29
    REGISTER_T5 = 30
    REGISTER_T6 = 31

    def __init__(self, bus: Bus):
        self._bus = bus
        self._pc: int = Ram.BASE
        self._regs: List[int] = [0] * 32
        self._regs[self.REGISTER_SP] = Ram.BASE + Ram.SIZE

    def dump_registers(self) -> None:
        logger.info(self._regs)

    def step(self) -> bool:
        inst = self.fetch()
        self._pc += 4
        self.execute(inst)
        self.dump_registers()
        if self._pc == 0:
            return False
        return True

    def fetch(self) -> int:
        return self._bus.load(self._pc, 32)

    def execute(self, inst: int) -> None:
        opcode: int = inst & 0x7F  # opcode in bits 6..0
        funct3: int = (inst >> 12) & 0x7  # funct3 in bits 14..12
        funct7: int = (inst >> 25) & 0x7F  # funct7 in bits 31..25

        self._regs[0] = 0
        # x0 hardwired to 0 at each cycle

        if opcode in [self.OP_LOAD, self.OP_IMM, self.OP_STORE]:
            if funct3 == self.FUNCT3_ADDI:
                self.exec_ADDI(inst)
            elif funct3 == self.FUNCT3_SLL:
                self.exec_SLLI(inst)
            elif funct3 == self.FUNCT3_SLTI:
                self.exec_SLTI(inst)
            elif funct3 == self.FUNCT3_SLTIU:
                self.exec_SLTIU(inst)
            elif funct3 == self.FUNCT3_XORI:
                self.exec_XORI(inst)
            elif funct3 == self.FUNCT3_SRL_SRA:
                if funct7 == self.FUNCT7_SRLI:
                    self.exec_SRLI(inst)
                elif funct7 == self.FUNCT7_SRAI:
                    self.exec_SRAI(inst)
                else:
                    pass
            elif funct3 == self.FUNCT3_ORI:
                self.exec_ORI(inst)
            elif funct3 == self.FUNCT3_ANDI:
                self.exec_ANDI(inst)
            else:
                pass
        else:
            logger.error(
                f"opcode:0x{opcode:02x}, funct3:0x{funct3:02x}, funct3:0x{funct7:02x}"
            )
            raise RuntimeError("Unknown instruction")

    def _load(self, addr: int, size: int) -> int:
        return self._bus.load(addr, size)

    def _store(self, addr: int, size: int, value: int) -> None:
        return self._bus.store(addr, size, value)

    @staticmethod
    def _rd(inst: int) -> int:
        return (inst >> 7) & 0x1F  # rd in bits 11..7

    @staticmethod
    def _rs1(inst: int) -> int:
        return (inst >> 15) & 0x1F  # rs1 in bits 19..15

    @staticmethod
    def _rs2(inst: int) -> int:
        return (inst >> 20) & 0x1F  # rs2 in bits 24..20

    @staticmethod
    def _imm_I(inst: int) -> int:
        # imm[11:0] = inst[31:20]
        return ((inst & 0xFFF00000)) >> 20

    @staticmethod
    def _imm_S(inst: int) -> int:
        # imm[11:5] = inst[31:25], imm[4:0] = inst[11:7]
        return ((inst & 0xFE000000) >> 20) | ((inst >> 7) & 0x1F)

    @staticmethod
    def _imm_B(inst: int) -> int:
        # imm[12|10:5|4:1|11] = inst[31|30:25|11:8|7]
        return (
            ((inst & 0x80000000) >> 19)
            | ((inst & 0x80) << 4)  # imm[11]
            | ((inst >> 20) & 0x7E0)  # imm[10:5]
            | ((inst >> 7) & 0x1E)
        )  # imm[4:1]

    @staticmethod
    def _imm_U(inst: int) -> int:
        # imm[31:12] = inst[31:12]
        return inst & 0xFFFFF999

    @staticmethod
    def _imm_J(inst: int) -> int:
        # imm[20|10:1|11|19:12] = inst[31|30:21|20|19:12]
        return (
            ((inst & 0x80000000) >> 11)
            | (inst & 0xFF000)  # imm[19:12]
            | ((inst >> 9) & 0x800)  # imm[11]
            | ((inst >> 20) & 0x7FE)
        )  # imm[10:1]

    def _shamt(self, inst: int) -> int:
        # shamt(shift amount) only required for immediate shift instructions
        # shamt[4:5] = imm[5:0]
        return self._imm_I(inst) & 0x1F  # TODO: 0x1f / 0x3f ?

    def exec_ADDI(self, inst: int) -> None:
        imm = self._imm_I(inst)
        self._regs[self._rd(inst)] = self._regs[self._rs1(inst)] + imm
        logger.debug("addi")

    def exec_SLTI(self, inst: int) -> None:
        imm = self._imm_I(inst)
        self._regs[self._rd(inst)] = 1 if (self._regs[self._rs1(inst)] < imm) else 0
        logger.debug("slti")

    def exec_SLTIU(self, inst: int) -> None:
        imm = self._imm_I(inst)
        self._regs[self._rd(inst)] = 1 if self._regs[self._rs1(inst)] < imm else 0
        logger.debug("sltiu")

    def exec_SRAI(self, inst: int) -> None:
        imm = self._imm_I(inst)
        self._regs[self._rd(inst)] = self._regs[self._rs1(inst)] >> imm
        logger.debug("srai")

    def exec_XORI(self, inst: int) -> None:
        imm = self._imm_I(inst)
        self._regs[self._rd(inst)] = self._regs[self._rs1(inst)] ^ imm
        logger.debug("xori")

    def exec_ORI(self, inst: int) -> None:
        imm = self._imm_I(inst)
        self._regs[self._rd(inst)] = self._regs[self._rs1(inst)] | imm
        logger.debug("ori")

    def exec_ANDI(self, inst: int) -> None:
        imm = self._imm_I(inst)
        self._regs[self._rd(inst)] = self._regs[self._rs1(inst)] & imm
        logger.debug("andi")

    def exec_SLLI(self, inst: int) -> None:
        self._regs[self._rd(inst)] = self._regs[self._rs1(inst)] << self._shamt(inst)
        logger.debug("slli")

    def exec_SRLI(self, inst: int) -> None:
        self._regs[self._rd(inst)] = self._regs[self._rs1(inst)] >> self._shamt(inst)
        logger.debug("srli")


def main(path: str) -> None:
    ram = Ram()
    bus = Bus(ram)
    cpu = Cpu(bus)

    ram.read_file(path)

    while cpu.step():
        pass


if __name__ == "__main__":
    main("a.out")
