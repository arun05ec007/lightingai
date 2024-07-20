import { CommonModule } from '@angular/common';
import { HttpClient } from '@angular/common/http';
import { Component } from '@angular/core';
import { FormsModule } from '@angular/forms';

@Component({
  selector: 'app-chat',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './chat.component.html',
  styleUrl: './chat.component.scss',
})
export class ChatComponent {
  chat: Array<{ role: string; message: string }> = [];
  Question: string = '';

  constructor(private http: HttpClient) {}

  pushMessage() {
    if (this.Question != '') {
      let ques = this.Question;
      this.Question = '';
      this.chat.push({ role: 'user', message: ques });
      this.chat.push({ role: 'bot', message: '...' });
      this.http
        .post<string>(
          'https://8000-01j10v4pkgxq5dx4v24zhwjffh.cloudspaces.litng.ai/Chatbot?Question=' +
            ques +
            '&File_name=abc',
          {}
        )
        .subscribe((res: string) => {
          this.chat.pop();
          this.chat.push({ role: 'bot', message: res[2] });
        });
    }
  }
}
