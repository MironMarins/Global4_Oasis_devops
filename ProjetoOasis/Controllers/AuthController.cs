using Microsoft.AspNetCore.Mvc;
using ProjetoOasis.Services;

namespace ProjetoOasis.Controllers
{

    [Route("api/[controller]")]
    [ApiController]
    public class AuthController : ControllerBase
    {


        private readonly TokenService _tokenService;

        public AuthController(TokenService tokenService)
        {
            _tokenService = tokenService;
        }


        [HttpPost("login")]
        public IActionResult Login([FromBody] LoginModel login)
        {
            if (login.Username == "filipe" && login.Password == "1234")
            {
                var token = _tokenService.GenerateToken(login.Username);
                return Ok(new { token });
            }

            return Unauthorized();
        }
    }

    public class LoginModel
    {
        public string Username { get; set; }
        public string Password { get; set; }
    }

    public class LoginResponse
    {
        public string Token { get; set; }
    }
}